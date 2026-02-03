
# To run code you need to run ~ % python /Users/ernestolopez/Downloads/Airport_vertices_optionB_worldmap_interactive.py

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button, TextBox
from networkx.algorithms.clique import find_cliques

# -------------------------------------------------------------------
# Files: by default expect airports.dat and routes.dat next to this .py
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
AIRPORTS_FILE = BASE_DIR / "airports.dat"
ROUTES_FILE   = BASE_DIR / "routes.dat"


def load_airports(path: Path):
    """
    Parses airports.dat (OpenFlights format).
    Returns:
      airports_by_id: {airport_id(int): attr_dict}
      code_to_id: {IATA/ICAO code(str): airport_id(int)}
    """
    airports_by_id = {}
    code_to_id = {}

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 14:
                continue

            (airport_id, name, city, country, iata, icao,
             lat, lon, *_rest) = row

            try:
                aid = int(airport_id)
            except ValueError:
                continue

            def norm(x: str):
                return None if (not x or x == r"\N") else x

            airports_by_id[aid] = {
                "name": name,
                "city": city,
                "country": country,
                "iata": norm(iata),
                "icao": norm(icao),
                "lat": float(lat) if lat and lat != r"\N" else None,
                "lon": float(lon) if lon and lon != r"\N" else None,
            }

            for code in (iata, icao):
                code = norm(code)
                if code:
                    code_to_id[code] = aid

    return airports_by_id, code_to_id


def build_airport_graph(
    airports_file: Path = AIRPORTS_FILE,
    routes_file: Path = ROUTES_FILE,
    directed: bool = True,
    aggregate_parallel_routes: bool = True,
):
    """
    Builds a graph where vertices are airports (from airports.dat)
    and edges are routes (from routes.dat).

    If aggregate_parallel_routes=True, multiple route records between the same (u,v)
    are merged with an edge attribute 'routes' counting how many, plus 'airlines'.
    """
    airports_by_id, code_to_id = load_airports(airports_file)

    G = nx.DiGraph() if directed else nx.Graph()

    # Add all airports as vertices
    for aid, attrs in airports_by_id.items():
        G.add_node(aid, **attrs)

    def resolve_airport_id(code: str, id_str: str):
        """Prefer numeric airport ID, otherwise fall back to IATA/ICAO code lookup."""
        if id_str and id_str != r"\N":
            try:
                return int(id_str)
            except ValueError:
                pass
        if code and code != r"\N":
            return code_to_id.get(code)
        return None

    # Add edges from routes.dat
    with open(routes_file, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue

            airline, airline_id, src_code, src_id, dst_code, dst_id = row[:6]

            u = resolve_airport_id(src_code, src_id)
            v = resolve_airport_id(dst_code, dst_id)

            # Skip routes that reference airports we couldn't resolve
            if u is None or v is None or u not in G or v not in G:
                continue

            if aggregate_parallel_routes:
                if G.has_edge(u, v):
                    G[u][v]["routes"] += 1
                    G[u][v]["airlines"].add(airline)
                else:
                    G.add_edge(u, v, routes=1, airlines={airline})
            else:
                G.add_edge(u, v, airline=airline)

    # Make airlines JSON/print-friendly
    if aggregate_parallel_routes:
        for _u, _v, data in G.edges(data=True):
            data["airlines"] = sorted(a for a in data["airlines"] if a and a != r"\N")

    return G


# -----------------------------
# OPTION B: Geographic plot
# -----------------------------
def draw_routes_geo(
    G: nx.DiGraph,
    top_n: int = 300,
    min_routes: int = 1,
    edge_alpha: float = 0.08,
    node_alpha: float = 0.85,
    out_png: str = "airport_graph_geo.png",
    show_labels: bool = False,
    world_map: bool = True,
):
    """
    Non-interactive geographic illustration (Option B) with an optional world map background.
    """
    # Keep only top airports by degree to avoid an unreadable plot
    top_nodes = [n for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]]
    H = G.subgraph(top_nodes).copy()

    # Drop nodes without coordinates
    coords: dict[int, tuple[float, float]] = {}
    for n in list(H.nodes):
        lat = H.nodes[n].get("lat")
        lon = H.nodes[n].get("lon")
        if lat is None or lon is None:
            H.remove_node(n)
        else:
            coords[n] = (float(lon), float(lat))

    if H.number_of_nodes() == 0:
        print("No nodes with coordinates to plot.")
        return H

    ax = None
    cartopy_ok = False

    if world_map:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            plt.figure(figsize=(18, 9))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_global()

            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)

            gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False

            cartopy_ok = True
        except Exception:
            cartopy_ok = False
            print("Cartopy not available; proceeding without world map background.")

    if ax is None:
        plt.figure(figsize=(18, 9))
        ax = plt.gca()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, linewidth=0.2, alpha=0.4)

    _plot_geo_on_ax(
        ax=ax,
        H=H,
        coords=coords,
        cartopy_ok=cartopy_ok,
        min_routes=min_routes,
        edge_alpha=edge_alpha,
        node_alpha=node_alpha,
        show_labels=show_labels,
        title=f"Routes {top_n} Airports (World Map)",
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")

    return H


# -----------------------------
# Interactive: filter by country
# -----------------------------
def draw_routes_geo_interactive(
    G: nx.DiGraph,
    top_n: int = 300,
    min_routes: int = 1,
    edge_alpha: float = 0.08,
    node_alpha: float = 0.85,
    show_labels: bool = False,
    world_map: bool = True,
):
    """
    Interactive geographic plot with a text box to filter airports by country.

    How it matches:
      - Type one or more countries, comma-separated.
      - Matching is case-insensitive and substring-based.
        Example: "United States, Canada" or "states" or "South".

    Filtering behavior:
      - Keeps airports whose country matches any typed term.
      - Then takes the top_n of those (by global degree) and plots only routes between them.
      - Clear the box (empty) to go back to the global top_n.
    """
    # Try Cartopy if requested
    cartopy_ok = False
    ccrs = None
    cfeature = None

    if world_map:
        try:
            import cartopy.crs as _ccrs
            import cartopy.feature as _cfeature

            cartopy_ok = True
            ccrs = _ccrs
            cfeature = _cfeature
        except Exception:
            cartopy_ok = False
            print("Cartopy not available; proceeding without world map background.")

    # Build figure/axes
    fig = plt.figure(figsize=(18, 9))
    if cartopy_ok:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)

    # Leave room at bottom for widgets
    fig.subplots_adjust(bottom=0.14)

    # Widgets
    axbox = fig.add_axes([0.16, 0.04, 0.62, 0.055])
    textbox = TextBox(axbox, "Countries:", initial="")

    ax_apply = fig.add_axes([0.80, 0.04, 0.09, 0.055])
    btn_apply = Button(ax_apply, "Apply")

    ax_clear = fig.add_axes([0.90, 0.04, 0.08, 0.055])
    btn_clear = Button(ax_clear, "Clear")

    state = {"last_query": ""}

    def parse_terms(q: str) -> list[str]:
        return [t.strip().lower() for t in (q or "").split(",") if t.strip()]

    def matches(country: str | None, terms: list[str]) -> bool:
        if not terms:
            return True
        c = (country or "").lower()
        return any(t in c for t in terms)

    def build_filtered_subgraph(q: str):
        terms = parse_terms(q)

        if terms:
            candidates = [n for n, attrs in G.nodes(data=True) if matches(attrs.get("country"), terms)]
            if not candidates:
                return nx.DiGraph(), {}, terms

            # Top-N within the filtered candidate set (using global degree)
            candidates_sorted = sorted(candidates, key=lambda n: G.degree(n), reverse=True)[:top_n]
            H = G.subgraph(candidates_sorted).copy()
        else:
            top_nodes = [n for n, d in sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]]
            H = G.subgraph(top_nodes).copy()

        # Drop nodes without coordinates
        coords: dict[int, tuple[float, float]] = {}
        for n in list(H.nodes):
            lat = H.nodes[n].get("lat")
            lon = H.nodes[n].get("lon")
            if lat is None or lon is None:
                H.remove_node(n)
            else:
                coords[n] = (float(lon), float(lat))

        return H, coords, terms

    def reset_basemap():
        ax.clear()
        if cartopy_ok:
            ax.set_global()
            ax.add_feature(cfeature.OCEAN)
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)

            gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4)
            gl.top_labels = False
            gl.right_labels = False
        else:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(True, linewidth=0.2, alpha=0.4)

    def redraw(q: str):
        state["last_query"] = q
        H, coords, terms = build_filtered_subgraph(q)

        reset_basemap()

        if H.number_of_nodes() == 0:
            ax.set_title("No airports matched that country filter.")
            fig.canvas.draw_idle()
            return

        filter_label = "All (top airports)" if not terms else " OR ".join(terms)
        _plot_geo_on_ax(
            ax=ax,
            H=H,
            coords=coords,
            cartopy_ok=cartopy_ok,
            min_routes=min_routes,
            edge_alpha=edge_alpha,
            node_alpha=node_alpha,
            show_labels=show_labels,
            title=f"Routes (filter by: {filter_label})  |  Nodes={H.number_of_nodes()} Edges={H.number_of_edges()}",
            cartopy_crs=ccrs,
        )
        fig.canvas.draw_idle()

    def on_submit(text):
        redraw(text)

    def on_apply(_event):
        redraw(textbox.text)

    def on_clear(_event):
        textbox.set_val("")
        redraw("")

    textbox.on_submit(on_submit)
    btn_apply.on_clicked(on_apply)
    btn_clear.on_clicked(on_clear)

    # Initial draw
    redraw("")

    print("Type countries (comma-separated) in the box and press Enter, or click Apply.")
    plt.show()


def _plot_geo_on_ax(
    ax,
    H: nx.DiGraph,
    coords: dict[int, tuple[float, float]],
    cartopy_ok: bool,
    min_routes: int,
    edge_alpha: float,
    node_alpha: float,
    show_labels: bool,
    title: str,
    cartopy_crs=None,
):
    """Internal helper: plots the given subgraph H on an existing axis."""
    # Draw edges
    for u, v, data in H.edges(data=True):
        if u not in coords or v not in coords:
            continue

        routes = data.get("routes", 1)
        if routes < min_routes:
            continue

        x1, y1 = coords[u]
        x2, y2 = coords[v]

        lw = 0.1 + 0.4 * math.log10(routes + 1)

        if cartopy_ok and cartopy_crs is not None:
            ax.plot([x1, x2], [y1, y2], linewidth=lw, alpha=edge_alpha, transform=cartopy_crs.Geodetic())
        else:
            ax.plot([x1, x2], [y1, y2], linewidth=lw, alpha=edge_alpha)

    # Draw nodes
    deg = dict(H.degree)
    xs = [coords[n][0] for n in H.nodes]
    ys = [coords[n][1] for n in H.nodes]
    sizes = [6 + 3 * math.sqrt(deg[n]) for n in H.nodes]

    if cartopy_ok and cartopy_crs is not None:
        ax.scatter(xs, ys, s=sizes, alpha=node_alpha, transform=cartopy_crs.PlateCarree())
    else:
        ax.scatter(xs, ys, s=sizes, alpha=node_alpha)

    # Optional labels (can clutter quickly)
    if show_labels:
        for n in H.nodes:
            code = H.nodes[n].get("iata") or H.nodes[n].get("icao")
            if not code:
                continue
            x, y = coords[n]
            if cartopy_ok and cartopy_crs is not None:
                ax.text(x, y, code, fontsize=6, alpha=0.9, transform=cartopy_crs.PlateCarree())
            else:
                ax.text(x, y, code, fontsize=6, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


# -----------------------------
# Extra helpers
# -----------------------------
def print_graph_type(G: nx.Graph):
    print("Graph type:", "DIRECTED" if G.is_directed() else "UNDIRECTED", f"({type(G).__name__})")


def print_maximum_clique(G: nx.Graph, label: str = "", limit: int | None = None):
    """
    Prints the maximum clique of G.
    If G is directed, this uses an undirected projection (G.to_undirected()).
    If limit is provided, it prints only the first `limit` clique members.
    """
    U = G.to_undirected() if G.is_directed() else G
    max_clique = max(find_cliques(U), key=len, default=[])

    print(f"\nMaximum clique {label}".rstrip())
    if G.is_directed():
        print("\n")  # computed on undirected projection
    print("Clique size:", len(max_clique))

    shown = max_clique if limit is None else max_clique[:limit]
    for aid in shown:
        info = G.nodes[aid]
        code = info.get("iata") or info.get("icao") or str(aid)
        print(f" - {code:>4}  {info.get('name')}, {info.get('country')}")

    if limit is not None and len(max_clique) > limit:
        print(f" ... and {len(max_clique) - limit} more")


if __name__ == "__main__":
    G = build_airport_graph(directed=True)

    print("\n")
    print_graph_type(G)

    print("Number of vertices (airports):", G.number_of_nodes())
    print("Number of edges (routes):", G.number_of_edges())

    # Example: top 10 airports by total degree (in+out)
    top10 = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 by degree:")
    for aid, deg in top10:
        info = G.nodes[aid]
        code = info.get("iata") or info.get("icao") or str(aid)
        print(f"{code:>4}  degree={deg:>5}  {info.get('name')}, {info.get('country')}")

    # Interactive geographic illustration
    draw_routes_geo_interactive(
        G,
        top_n=300,
        min_routes=1,
        show_labels=False,
        world_map=True,
    )
