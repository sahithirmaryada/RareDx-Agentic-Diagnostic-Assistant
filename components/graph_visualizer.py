import json
import re
import hashlib
from pathlib import Path

from streamlit.components.v1 import html

VIS_JS_PATH = Path(__file__).resolve().parents[1] / "lib" / "vis-9.1.2" / "vis-network.min.js"

RELATIONSHIP_PATTERN = re.compile(
    r"(?P<src_type>\w+)\[(?P<src_label>[^\]]+)\]\s*-\[:(?P<rel>[^\]]+)\]->\s*(?P<dst_type>\w+)\[(?P<dst_label>[^\]]+)\]",
    re.IGNORECASE,
)

NODE_COLORS = {
    "Disease": {"background": "#b42318", "border": "#7a271a"},
    "Symptom": {"background": "#2563eb", "border": "#1d4ed8"},
    "Gene": {"background": "#159947", "border": "#0f6b32"},
}

NODE_SHAPES = {
    "Disease": "box",
    "Symptom": "ellipse",
    "Gene": "diamond",
}

DEFAULT_NODE_COLOR = {"background": "#64748b", "border": "#475569"}
EDGE_COLOR = "#95a5a6"


def _load_vis_js():
    if not VIS_JS_PATH.exists():
        raise FileNotFoundError(f"vis.js file not found at {VIS_JS_PATH}")

    with open(VIS_JS_PATH, "r", encoding="utf-8") as fp:
        return fp.read()


def parse_graph_paths(graph_paths):
    node_map = {}
    nodes = []
    edges = []
    edge_keys = set()
    next_id = 1

    def _add_node(node_type, label):
        nonlocal next_id
        node_type = normalize_node_type(node_type)
        label = normalize_label(label)
        key = f"{node_type}:{label.casefold()}"
        if key in node_map:
            return node_map[key]

        node_id = next_id
        next_id += 1
        node_map[key] = node_id

        nodes.append(
            {
                "id": node_id,
                "label": label,
                "title": f"{node_type}: {label}",
                "group": node_type,
                "color": NODE_COLORS.get(node_type, DEFAULT_NODE_COLOR),
                "shape": NODE_SHAPES.get(node_type, "ellipse"),
                "font": {"color": "#ffffff", "strokeWidth": 0, "size": 14},
                "margin": 10,
            }
        )
        return node_id

    parsed_paths = 0
    skipped_paths = []

    for path in normalize_graph_paths(graph_paths):
        if not isinstance(path, str):
            continue
        match = RELATIONSHIP_PATTERN.search(path)
        if not match:
            skipped_paths.append(path)
            continue

        src_type = normalize_node_type(match.group("src_type"))
        src_label = normalize_label(match.group("src_label"))
        rel = normalize_label(match.group("rel"))
        dst_type = normalize_node_type(match.group("dst_type"))
        dst_label = normalize_label(match.group("dst_label"))

        src_id = _add_node(src_type, src_label)
        dst_id = _add_node(dst_type, dst_label)
        edge_key = (src_id, dst_id, rel)
        if edge_key in edge_keys:
            continue
        edge_keys.add(edge_key)

        edges.append(
            {
                "from": src_id,
                "to": dst_id,
                "label": format_relationship(rel),
                "color": {"color": EDGE_COLOR},
                "arrows": "to",
                "font": {"align": "middle", "color": "#334155", "size": 11},
                "smooth": {"enabled": True, "type": "curvedCW"},
            }
        )
        parsed_paths += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "parsed_path_count": parsed_paths,
        "skipped_path_count": len(skipped_paths),
        "skipped_paths": skipped_paths,
    }


def _parse_graph_paths(graph_paths):
    return parse_graph_paths(graph_paths)


def normalize_graph_paths(graph_paths):
    if graph_paths is None:
        return []
    if isinstance(graph_paths, str):
        return [graph_paths]
    if isinstance(graph_paths, (list, tuple, set)):
        paths = []
        for item in graph_paths:
            if isinstance(item, (list, tuple, set)):
                paths.extend(normalize_graph_paths(item))
            else:
                paths.append(item)
        return paths
    return []


def normalize_node_type(node_type):
    normalized = re.sub(r"[^a-zA-Z0-9]+", "", str(node_type or "")).strip().lower()
    return {
        "disease": "Disease",
        "disorder": "Disease",
        "symptom": "Symptom",
        "hpo": "Symptom",
        "phenotype": "Symptom",
        "gene": "Gene",
    }.get(normalized, normalized.title() if normalized else "Unknown")


def normalize_label(label):
    return re.sub(r"\s+", " ", str(label or "")).strip()


def format_relationship(relationship):
    words = normalize_label(relationship).replace("_", " ").split()
    return " ".join(word.capitalize() for word in words)


def graph_dom_id(graph_data, graph_id=None):
    raw_id = graph_id or hashlib.sha1(
        json.dumps(graph_data, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(raw_id)).strip("-")
    return f"neo4j-graph-{safe_id or 'default'}"


def render_fallback(message, height=180):
    html(
        f"""
        <div style="height:{height}px; display:flex; align-items:center; justify-content:center;
                    border:1px solid #d0d7de; border-radius:8px; background:#ffffff;
                    color:#475569; font:14px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;
                    text-align:center; padding:18px;">
            {message}
        </div>
        """,
        height=height,
    )


def render_graph_from_paths(graph_paths, height=520, graph_id=None):
    graph_data = parse_graph_paths(graph_paths)

    if not graph_data["nodes"] or not graph_data["edges"]:
        render_fallback("No Neo4j graph paths are available for this candidate.", height=180)
        return

    try:
        vis_js = _load_vis_js()
    except FileNotFoundError:
        render_fallback("Graph visualization library is missing from lib/vis-9.1.2.", height=180)
        return

    escaped_data = json.dumps(graph_data)
    container_id = graph_dom_id(graph_data, graph_id)

    html_template = """
    <style>
        #__GRAPH_ID___wrap {
            border: 1px solid #d0d7de;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        #__GRAPH_ID__ {
            height: __GRAPH_HEIGHT__px;
            background: #ffffff;
        }
        #__GRAPH_ID___legend {
            display: flex;
            gap: 14px;
            align-items: center;
            padding: 10px 12px;
            border-bottom: 1px solid #e5e7eb;
            color: #334155;
            font-size: 12px;
        }
        .graph-legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            white-space: nowrap;
        }
        .graph-legend-dot {
            width: 11px;
            height: 11px;
            border-radius: 999px;
            display: inline-block;
        }
    </style>
    <div id="__GRAPH_ID___wrap">
        <div id="__GRAPH_ID___legend">
            <span class="graph-legend-item"><span class="graph-legend-dot" style="background:#b42318;"></span>Disease</span>
            <span class="graph-legend-item"><span class="graph-legend-dot" style="background:#2563eb;"></span>Symptom</span>
            <span class="graph-legend-item"><span class="graph-legend-dot" style="background:#159947;"></span>Gene</span>
        </div>
        <div id="__GRAPH_ID__"></div>
    </div>
    <script type="text/javascript">
    __VIS_JS__
    </script>
    <script type="text/javascript">
        const graphData = __GRAPH_DATA__;
        const container = document.getElementById('__GRAPH_ID__');
        const data = {
            nodes: new vis.DataSet(graphData.nodes),
            edges: new vis.DataSet(graphData.edges),
        };
        const options = {
            physics: {
                enabled: true,
                barnesHut: {
                    gravitationalConstant: -2500,
                    centralGravity: 0.3,
                    springLength: 190,
                    springConstant: 0.04,
                    damping: 0.09,
                },
                stabilization: {
                    enabled: true,
                    iterations: 250,
                    updateInterval: 25,
                },
            },
            nodes: {
                borderWidth: 2,
                borderWidthSelected: 4,
                font: { size: 14, face: 'Helvetica' },
                scaling: { label: { enabled: true } },
                shadow: { enabled: true, color: 'rgba(15,23,42,0.16)', size: 8, x: 1, y: 2 },
            },
            edges: {
                color: '#95a5a6',
                smooth: true,
                arrows: {
                    to: { enabled: true, scaleFactor: 0.6 },
                },
            },
            interaction: {
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true,
            },
            layout: {
                improvedLayout: true,
            },
            manipulation: {
                enabled: false,
            },
        };

        const network = new vis.Network(container, data, options);
        network.on('selectNode', function(params) {
            const node = graphData.nodes.find(n => n.id === params.nodes[0]);
            if (node) {
                const detail = `Type: ${node.title.split(':')[0]}\nName: ${node.label}`;
                console.log(detail);
            }
        });
    </script>
    """

    graph_height = max(int(height) - 44, 220)
    html_content = (
        html_template.replace("__GRAPH_HEIGHT__", str(graph_height))
        .replace("__GRAPH_ID__", container_id)
        .replace("__VIS_JS__", vis_js)
        .replace("__GRAPH_DATA__", escaped_data)
    )
    html(html_content, height=height)
