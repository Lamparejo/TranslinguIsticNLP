"""Streamlit dashboard visualising knowledge graph metrics."""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt  # type: ignore[import-not-found]
import polars as pl  # type: ignore[import-not-found]
import streamlit as st  # type: ignore[import-not-found]
from streamlit.components.v1 import html as components_html  # type: ignore[attr-defined]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_INFERENCE_IMPORT_ERROR: str | None = None
_TRAIN_IMPORT_ERROR: str | None = None
_CONFIG_IMPORT_ERROR: str | None = None

try:  # Optional dependencies for GNN inference
    from src.gnn.inference import (
        load_inference_artifacts,
        predict_link_probability,
        recommend_targets,
    )
except Exception:  # pragma: no cover - keep dashboard running even without torch
    load_inference_artifacts = None  # type: ignore[assignment]
    predict_link_probability = None  # type: ignore[assignment]
    recommend_targets = None  # type: ignore[assignment]
    _INFERENCE_IMPORT_ERROR = traceback.format_exc()
else:
    _INFERENCE_IMPORT_ERROR = None

try:  # Optional training utilities
    from src.pipelines.train_gnn import train_from_config
except Exception:  # pragma: no cover - training stays optional
    train_from_config = None  # type: ignore[assignment]
    _TRAIN_IMPORT_ERROR = traceback.format_exc()
else:
    _TRAIN_IMPORT_ERROR = None

try:  # Config loader (optional convenience)
    from src.utils.config import load_config as load_pipeline_config
except Exception:
    load_pipeline_config = None  # type: ignore[assignment]
    _CONFIG_IMPORT_ERROR = traceback.format_exc()
else:
    _CONFIG_IMPORT_ERROR = None

DEFAULT_METRICS_PATH = Path("artifacts/dashboard_metrics.json")
FALLBACK_METRICS_PATH = Path("artifacts/example_dashboard_metrics.json")
DEFAULT_GRAPH_SNAPSHOT_PATH = Path("artifacts/graph_snapshot.json")
DEFAULT_GNN_METRICS_PATH = Path("artifacts/gnn_training_metrics.json")
DEFAULT_GNN_MODEL_PATH = Path("artifacts/gnn_link_predictor.pt")
DEFAULT_PYG_DATASET_PATH = Path("artifacts/pyg_graph.pt")


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _ensure_dict(json.load(handle))


@st.cache_data(show_spinner=False)
def load_graph_snapshot(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _ensure_dict(json.load(handle))


@st.cache_data(show_spinner=False)
def load_gnn_metrics_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _ensure_dict(json.load(handle))


@st.cache_resource(show_spinner=False)
def get_inference_artifacts(model_path: Path, dataset_path: Path, device: str = "cpu"):
    if load_inference_artifacts is None:
        return None
    return load_inference_artifacts(model_path, dataset_path, device=device)


def resolve_metrics_paths() -> Tuple[Path, bool]:
    if DEFAULT_METRICS_PATH.exists():
        return DEFAULT_METRICS_PATH, False
    if FALLBACK_METRICS_PATH.exists():
        return FALLBACK_METRICS_PATH, True
    raise FileNotFoundError(
        "No metrics file found. Run the pipeline or provide a metrics JSON via the uploader."
    )

def _ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_list_of_dicts(value: Any) -> list[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _format_large_int(value: Any) -> str:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return "–"
    return f"{integer:,}".replace(",", ".")


def _format_duration(seconds: Any) -> str:
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return "–"
    if total_seconds <= 0:
        return "–"
    minutes, sec = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _first_available(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return default


def _resolve_graph_snapshot_path(metadata: Dict[str, Any], metrics_path: Path) -> Path | None:
    candidates: list[Path] = []
    if isinstance(metadata, dict):
        declared = metadata.get("graph_snapshot_path")
        if isinstance(declared, str) and declared:
            declared_path = Path(declared)
            if not declared_path.exists():
                declared_path = (metrics_path.parent / declared).resolve()
            candidates.append(declared_path)
    candidates.append(DEFAULT_GRAPH_SNAPSHOT_PATH)
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _render_training_sidebar(metadata: Dict[str, Any], gnn_metrics: Dict[str, Any]) -> None:
    st.sidebar.header("Treinamento do GNN")

    if train_from_config is None:
        st.sidebar.info(
            "Dependências de treinamento (torch/torch-geometric) indisponíveis. Instale-as para reexecutar o GNN por aqui."
        )
        if _TRAIN_IMPORT_ERROR:
            with st.sidebar.expander("Detalhes técnicos", expanded=False):
                st.sidebar.code(_TRAIN_IMPORT_ERROR, language="text")
        return

    config_path_default = str(_first_available(metadata.get("config_path"), "config/pipeline.yaml"))
    config_data: Dict[str, Any] = {}
    config_error: str | None = None
    if load_pipeline_config is not None:
        try:
            config_data = load_pipeline_config(config_path_default)
        except Exception as exc:  # pragma: no cover - informational only
            config_error = str(exc)
    elif _CONFIG_IMPORT_ERROR:
        config_error = "Leitor de configuração indisponível no ambiente atual."

    gnn_defaults = _ensure_dict(config_data.get("gnn"))
    gnn_metrics_config = _ensure_dict(gnn_metrics.get("config"))
    gnn_meta = _ensure_dict(gnn_metrics.get("metadata"))
    model_defaults = _ensure_dict(gnn_metrics_config.get("model"))
    training_defaults = _ensure_dict(gnn_metrics_config.get("training"))

    default_dataset = str(
        _first_available(
            gnn_defaults.get("dataset_path"),
            gnn_meta.get("dataset_path"),
            metadata.get("gnn_dataset_path"),
            DEFAULT_PYG_DATASET_PATH.as_posix(),
        )
    )
    default_metrics_path = str(
        _first_available(
            gnn_defaults.get("metrics_path"),
            gnn_meta.get("metrics_path"),
            metadata.get("gnn_metrics_path"),
            DEFAULT_GNN_METRICS_PATH.as_posix(),
        )
    )
    default_model_path = str(
        _first_available(
            gnn_defaults.get("model_artifact_path"),
            gnn_meta.get("model_artifact_path"),
            metadata.get("gnn_model_path"),
            DEFAULT_GNN_MODEL_PATH.as_posix(),
        )
    )

    hidden_default = _safe_int(
        _first_available(gnn_defaults.get("hidden_channels"), model_defaults.get("hidden_channels"), default=128),
        128,
    )
    out_default = _safe_int(
        _first_available(gnn_defaults.get("out_channels"), model_defaults.get("out_channels"), default=64),
        64,
    )
    dropout_default = _safe_float(
        _first_available(gnn_defaults.get("dropout"), model_defaults.get("dropout"), default=0.3),
        0.3,
    )
    epochs_default = _safe_int(
        _first_available(gnn_defaults.get("epochs"), training_defaults.get("epochs"), default=10),
        10,
    )
    lr_default = _safe_float(
        _first_available(gnn_defaults.get("learning_rate"), training_defaults.get("learning_rate"), default=1e-3),
        1e-3,
    )
    weight_decay_default = _safe_float(
        _first_available(gnn_defaults.get("weight_decay"), training_defaults.get("weight_decay"), default=1e-4),
        1e-4,
    )
    device_default = str(
        _first_available(gnn_defaults.get("device"), training_defaults.get("device"), default="cpu")
    )
    val_ratio_default = _safe_float(
        _first_available(gnn_defaults.get("val_ratio"), gnn_meta.get("val_ratio"), default=0.1),
        0.1,
    )
    test_ratio_default = _safe_float(
        _first_available(gnn_defaults.get("test_ratio"), gnn_meta.get("test_ratio"), default=0.1),
        0.1,
    )

    with st.sidebar.form("train_gnn_form"):
        config_path_input = st.text_input("Arquivo de configuração", config_path_default)
        dataset_path_input = st.text_input("Snapshot PyG (.pt)", default_dataset)
        metrics_path_input = st.text_input("Arquivo de métricas", default_metrics_path)
        model_path_input = st.text_input("Checkpoint do modelo", default_model_path)

        hidden_channels = st.number_input(
            "Hidden channels",
            min_value=16,
            max_value=4096,
            step=16,
            value=hidden_default,
        )
        out_channels = st.number_input(
            "Out channels",
            min_value=8,
            max_value=2048,
            step=8,
            value=out_default,
        )
        dropout = st.number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.9,
            step=0.05,
            format="%.2f",
            value=float(dropout_default),
        )
        epochs = st.number_input("Épocas", min_value=1, max_value=500, value=epochs_default, step=1)
        learning_rate = st.number_input(
            "Learning rate",
            min_value=1e-5,
            max_value=1.0,
            value=float(lr_default),
            step=1e-4,
            format="%.5f",
        )
        weight_decay = st.number_input(
            "Weight decay",
            min_value=0.0,
            max_value=1.0,
            value=float(weight_decay_default),
            step=1e-4,
            format="%.5f",
        )

        device_options = [device_default] if device_default not in {"cpu", "cuda"} else []
        device_options.extend([option for option in ("cpu", "cuda") if option not in device_options])
        device_choice = st.selectbox("Dispositivo", device_options, index=0)

        val_ratio = st.number_input(
            "Val ratio",
            min_value=0.0,
            max_value=0.5,
            value=float(val_ratio_default),
            step=0.05,
            format="%.2f",
        )
        test_ratio = st.number_input(
            "Test ratio",
            min_value=0.0,
            max_value=0.5,
            value=float(test_ratio_default),
            step=0.05,
            format="%.2f",
        )

        submitted = st.form_submit_button("Treinar novamente")

    if config_error:
        st.sidebar.warning(f"Configuração padrão não pôde ser carregada: {config_error}")

    dataset_exists = Path(dataset_path_input).exists()
    if dataset_exists:
        st.sidebar.caption(f"Snapshot detectado em `{dataset_path_input}`.")
    else:
        st.sidebar.info(
            "Snapshot PyG não encontrado. Execute o pipeline ou informe um caminho válido antes de iniciar o treino."
        )

    if submitted:
        if val_ratio + test_ratio >= 0.9:
            st.sidebar.error("A soma de Val/Test deve ser menor que 0.9 para manter parte dos dados de treino.")
            return

        overrides: Dict[str, Any] = {
            "hidden_channels": int(hidden_channels),
            "out_channels": int(out_channels),
            "dropout": float(dropout),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "device": device_choice,
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
        }

        config_path_final = config_path_input.strip() or config_path_default
        if dataset_path_input.strip():
            overrides["dataset_path"] = dataset_path_input.strip()
        if metrics_path_input.strip():
            overrides["metrics_path"] = metrics_path_input.strip()
        if model_path_input.strip():
            overrides["model_artifact_path"] = model_path_input.strip()

        with st.spinner("Treinando o GNN... isso pode levar alguns minutos"):
            try:
                train_from_config(config_path_final, overrides=overrides)
            except Exception as exc:  # pragma: no cover - surfaced to UI
                st.sidebar.error(f"Falha ao treinar o GNN: {exc}")
                return

        st.sidebar.success("Treino concluído! Atualizando métricas...")
        load_gnn_metrics_file.clear()  # type: ignore[attr-defined]
        get_inference_artifacts.clear()  # type: ignore[attr-defined]
        st.rerun()  # type: ignore[attr-defined]


def _render_graph(snapshot_path: Path) -> None:
    try:
        graph_data = load_graph_snapshot(snapshot_path)
    except FileNotFoundError:
        st.warning(f"Snapshot de grafo não encontrado em {snapshot_path}.")
        return
    except json.JSONDecodeError as exc:
        st.error(f"Não foi possível interpretar o snapshot do grafo: {exc}")
        return

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not nodes:
        st.write("O snapshot não contém nós para exibição.")
        return

    try:
        from pyvis.network import Network  # type: ignore[import-not-found]
    except ImportError:
        st.error(
            "Dependência `pyvis` ausente. Instale-a com `pip install pyvis` para visualizar o grafo."
        )
        return

    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#222222")  # type: ignore[arg-type]
    net.barnes_hut()

    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        net.add_node(
            node_id,
            label=node.get("label", node_id),
            title=_format_node_tooltip(node),
            value=node.get("mention_count", 1) or 1,
            group=node.get("type", "Entity"),
        )

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not source or not target:
            continue
        net.add_edge(
            source,
            target,
            value=edge.get("weight", 1.0) or 1.0,
            title=_format_edge_tooltip(edge),
        )

    html_content = net.generate_html(notebook=False)
    components_html(html_content, height=670)

    with st.expander("Dados do grafo", expanded=False):
        st.json(graph_data, expanded=False)


def _format_node_tooltip(node: Dict[str, Any]) -> str:
    label = node.get("label", node.get("id", "Entity"))
    entity_type = node.get("type", "?")
    mentions = node.get("mention_count", 0)
    return f"<strong>{label}</strong><br/>Tipo: {entity_type}<br/>Menções: {mentions}"


def _format_edge_tooltip(edge: Dict[str, Any]) -> str:
    relation_type = edge.get("relation_type", "MENTIONED_TOGETHER")
    weight = edge.get("weight", 1.0)
    sentence = edge.get("sentence")
    published = edge.get("published_at")
    tooltip = f"Tipo: {relation_type}<br/>Peso: {weight}"
    if sentence:
        tooltip += f"<br/>Frase: {sentence}"
    if published:
        tooltip += f"<br/>Publicado em: {published}"
    return tooltip


def main() -> None:
    st.set_page_config(page_title="Translinguistic Knowledge Graph Dashboard", layout="wide")
    st.title("Translinguistic Knowledge Graph Insights")
    st.caption(
        "Monitor multilingual entity extraction, graph construction, and GNN-readiness at a glance."
    )

    metrics_path, is_fallback = resolve_metrics_paths()
    metrics = load_metrics(metrics_path)

    with st.expander("Upload custom metrics", expanded=False):
        uploaded_file = st.file_uploader("Choose a JSON file exported by the pipeline")
        if uploaded_file:
            metrics = _ensure_dict(json.loads(uploaded_file.read()))
            metrics_path = Path(uploaded_file.name)
            is_fallback = False

    metadata = _ensure_dict(metrics.get("metadata"))
    entity_summary = _ensure_dict(metrics.get("entity_summary"))
    relation_summary = _ensure_dict(metrics.get("relation_summary"))
    articles_summary = _ensure_dict(metrics.get("articles"))
    language_distribution = _ensure_dict(metrics.get("language_distribution"))
    timeline = _ensure_dict(metrics.get("timeline"))
    daily_timeline = _ensure_list_of_dicts(timeline.get("daily", []))
    gnn_metrics_path_str = metadata.get("gnn_metrics_path") if metadata else None
    gnn_model_path_str = metadata.get("gnn_model_path") if metadata else None
    gnn_dataset_path_str = metadata.get("gnn_dataset_path") if metadata else None
    graph_snapshot_path = _resolve_graph_snapshot_path(metadata, metrics_path)

    info_cols = st.columns(4)
    info_cols[0].metric("Entities", metadata.get("num_entities", 0))
    info_cols[1].metric("Relations", metadata.get("num_relations", 0))
    info_cols[2].metric(
        "Avg mentions / entity",
        f"{entity_summary.get('avg_mentions_per_entity', 0.0):.2f}",
    )
    info_cols[3].metric(
        "Graph density",
        f"{relation_summary.get('graph_density', 0.0):.3f}",
    )

    if is_fallback:
        st.info(
            "Displaying example metrics. Run the pipeline to refresh `artifacts/dashboard_metrics.json`."
        )

    graph_summary = _ensure_dict(metrics.get("graph_summary"))
    if graph_summary:
        summary_cols = st.columns(4)
        summary_cols[0].metric(
            "Avg degree",
            f"{graph_summary.get('average_degree', 0.0):.2f}",
        )
        summary_cols[1].metric(
            "Max degree",
            int(graph_summary.get("max_degree", 0)),
        )
        summary_cols[2].metric(
            "Components",
            int(graph_summary.get("connected_components", 0)),
        )
        summary_cols[3].metric(
            "Largest component",
            int(graph_summary.get("largest_component_size", 0)),
        )

        advanced_cols = st.columns(4)
        advanced_cols[0].metric(
            "Median degree",
            f"{graph_summary.get('median_degree', 0.0):.2f}",
        )
        advanced_cols[1].metric(
            "Degree P90",
            f"{graph_summary.get('degree_percentile_90', 0.0):.2f}",
        )
        advanced_cols[2].metric(
            "Isolated nodes",
            _format_large_int(graph_summary.get("isolated_nodes", 0)),
        )
        advanced_cols[3].metric(
            "Avg clustering",
            f"{graph_summary.get('average_clustering_coefficient', 0.0):.3f}",
        )

    st.subheader("Entity Composition")
    entity_counts = _ensure_dict(entity_summary.get("entity_type_counts", {}))
    entity_df = pl.DataFrame(
        {"Entity Type": list(entity_counts.keys()), "Count": list(entity_counts.values())}
    )
    if not entity_df.is_empty():
        chart = (
            alt.Chart(data=entity_df.to_pandas())
            .mark_bar()
            .encode(
                x=alt.X("Entity Type", sort="-y"),
                y="Count",
                tooltip=["Entity Type", "Count"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No entity information available.")

    col_left, col_right = st.columns(2)

    relation_counts = _ensure_dict(relation_summary.get("relation_type_counts", {}))
    relation_df = pl.DataFrame(
        {
            "Relation Type": list(relation_counts.keys()),
            "Count": list(relation_counts.values()),
        }
    )
    with col_left:
        st.markdown("### Relation Types")
        if not relation_df.is_empty():
            relation_chart = (
                alt.Chart(data=relation_df.to_pandas())
                .mark_arc(innerRadius=50)
                .encode(theta="Count", color="Relation Type", tooltip=["Relation Type", "Count"])
            )
            st.altair_chart(relation_chart, use_container_width=True)
        else:
            st.write("No relation information available.")

    language_counts = language_distribution
    language_df = pl.DataFrame(
        {"Language": list(language_counts.keys()), "Mentions": list(language_counts.values())}
    )
    with col_right:
        st.markdown("### Language Coverage")
        if not language_df.is_empty():
            language_chart = (
                alt.Chart(data=language_df.to_pandas())
                .mark_bar()
                .encode(
                    x=alt.X("Mentions", title="Mentions", sort="-x"),
                    y=alt.Y("Language", sort="-x"),
                    tooltip=["Language", "Mentions"],
                )
            )
            st.altair_chart(language_chart, use_container_width=True)
        else:
            st.write("No language information available.")

    relation_stats_cols = st.columns(4)
    relation_stats_cols[0].metric(
        "Peso médio",
        f"{relation_summary.get('avg_relation_weight', 0.0):.2f}",
    )
    relation_stats_cols[1].metric(
        "Peso mediano",
        f"{relation_summary.get('median_relation_weight', 0.0):.2f}",
    )
    relation_stats_cols[2].metric(
        "Peso mínimo",
        f"{relation_summary.get('min_relation_weight', 0.0):.2f}",
    )
    relation_stats_cols[3].metric(
        "Desvio padrão",
        f"{relation_summary.get('std_relation_weight', 0.0):.3f}",
    )

    st.subheader("Top Entities by Degree")
    top_entities = _ensure_list_of_dicts(metrics.get("top_entities_by_degree", []))
    if top_entities:
        top_entities_df = pl.DataFrame(top_entities)
        top_entities_view = (
            top_entities_df.select(["name", "entity_type", "degree"]).rename(
                {"name": "Entity", "entity_type": "Type", "degree": "Degree"}
            )
        )
        st.dataframe(top_entities_view.to_pandas(), use_container_width=True)
    else:
        st.write("Degree centrality not available.")

    st.subheader("Temporal Coverage")
    time_span = _ensure_dict(articles_summary.get("time_span", {}))
    coverage_cols = st.columns(3)
    coverage_cols[0].metric("Earliest", time_span.get("start", "–"))
    coverage_cols[1].metric("Latest", time_span.get("end", "–"))
    coverage_cols[2].metric(
        "Active days",
        timeline.get("total_days", len(daily_timeline)),
    )

    timeline_stats_cols = st.columns(2)
    timeline_stats_cols[0].metric(
        "Pico relações/dia",
        timeline.get("peak_relations", 0),
    )
    timeline_stats_cols[1].metric(
        "Pico entidades/dia",
        timeline.get("peak_entities", 0),
    )

    st.subheader("Timeline of Activity")
    timeline_df = pl.DataFrame(daily_timeline)
    if not timeline_df.is_empty():
        timeline_df = (
            timeline_df.with_columns(
                pl.col("date").str.to_datetime(strict=False, time_unit="us")
            )
            .filter(pl.col("date").is_not_null())
            .sort("date")
        )
        if not timeline_df.is_empty():
            timeline_long = timeline_df.melt(
                id_vars=["date"],
                value_vars=["relations", "unique_entities"],
                variable_name="Metric",
                value_name="Count",
            )
            activity_chart = (
                alt.Chart(data=timeline_long.to_pandas())
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("Count:Q", title="Count"),
                    color=alt.Color("Metric:N", title="Metric"),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("Metric:N", title="Metric"),
                        alt.Tooltip("Count:Q", title="Count"),
                    ],
                )
            )
            st.altair_chart(activity_chart, use_container_width=True)
        else:
            st.write("Timeline data could not be parsed.")
    else:
        st.write("No timeline data available.")

    st.subheader("GNN Training Metrics")
    gnn_metrics_path = Path(gnn_metrics_path_str) if gnn_metrics_path_str else DEFAULT_GNN_METRICS_PATH
    gnn_metrics: Dict[str, Any] = {}
    if gnn_metrics_path.exists():
        try:
            gnn_metrics = load_gnn_metrics_file(gnn_metrics_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            st.warning(f"Não foi possível carregar métricas de treinamento do GNN ({exc}).")
    else:
        st.info("Nenhum registro de treinamento encontrado. Execute `scripts/train_gnn.py` para gerar métricas.")

    _render_training_sidebar(metadata, gnn_metrics)

    if gnn_metrics:
        gnn_meta = _ensure_dict(gnn_metrics.get("metadata"))
        gnn_metrics_config = _ensure_dict(gnn_metrics.get("config"))
        best_epoch = _ensure_dict(gnn_metrics.get("best_epoch"))
        final_epoch = _ensure_dict(gnn_metrics.get("final_epoch"))
        training_cols = st.columns(4)
        training_cols[0].metric("Epochs", gnn_meta.get("epochs", 0))
        training_cols[1].metric(
            "Best val AUC",
            f"{best_epoch.get('val_auc', 0.0):.3f}" if best_epoch else "–",
        )
        training_cols[2].metric(
            "Best val AP",
            f"{best_epoch.get('val_ap', 0.0):.3f}" if best_epoch else "–",
        )
        training_cols[3].metric(
            "Final test AUC",
            f"{final_epoch.get('test_auc', 0.0):.3f}" if final_epoch else "–",
        )

        meta_cols = st.columns(4)
        meta_cols[0].metric("Duração", _format_duration(gnn_meta.get("training_duration_seconds")))
        meta_cols[1].metric(
            "Parâmetros (treináveis)",
            _format_large_int(gnn_meta.get("trainable_parameters")),
        )
        meta_cols[2].metric(
            "Loss médio",
            f"{gnn_meta.get('avg_epoch_loss', 0.0):.4f}",
        )
        meta_cols[3].metric(
            "Val/Test",
            f"{gnn_meta.get('val_ratio', 0.0):.2f} / {gnn_meta.get('test_ratio', 0.0):.2f}",
        )

        gnn_history = _ensure_list_of_dicts(gnn_metrics.get("history"))
        history_df = pl.DataFrame(gnn_history) if gnn_history else pl.DataFrame([])
        if not history_df.is_empty() and "epoch" in history_df.columns:
            metric_fields = [
                field
                for field in ["val_auc", "test_auc", "val_ap", "test_ap", "loss"]
                if field in history_df.columns
            ]
            if metric_fields:
                history_long = (
                    history_df.select(["epoch", *metric_fields])
                    .melt(id_vars=["epoch"], value_vars=metric_fields, variable_name="Metric", value_name="Value")
                    .to_pandas()
                )
                training_chart = (
                    alt.Chart(history_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("epoch:Q", title="Epoch"),
                        y=alt.Y("Value:Q", title="Metric value"),
                        color=alt.Color("Metric:N", title="Metric"),
                        tooltip=["epoch", "Metric", alt.Tooltip("Value", format=".4f")],
                    )
                )
                st.altair_chart(training_chart, use_container_width=True)
        else:
            st.write("Histórico de treinamento indisponível.")

        with st.expander("Hiperparâmetros utilizados", expanded=False):
            st.json(
                {
                    "config": gnn_metrics_config,
                    "divisao": {
                        "val_ratio": gnn_meta.get("val_ratio"),
                        "test_ratio": gnn_meta.get("test_ratio"),
                    },
                    "overrides": gnn_meta.get("overrides"),
                    "artefatos": {
                        "dataset": gnn_meta.get("dataset_path"),
                        "metrics": gnn_meta.get("metrics_path"),
                        "modelo": gnn_meta.get("model_artifact_path"),
                    },
                }
            )

    st.subheader("GNN Link Prediction Playground")
    gnn_model_path = Path(gnn_model_path_str) if gnn_model_path_str else DEFAULT_GNN_MODEL_PATH
    gnn_dataset_path = Path(gnn_dataset_path_str) if gnn_dataset_path_str else DEFAULT_PYG_DATASET_PATH
    if load_inference_artifacts is None:
        st.info(
            "Dependências necessárias (torch/torch-geometric) não estão instaladas ou houve uma falha ao carregá-las; o playground está desativado."
        )
        if _INFERENCE_IMPORT_ERROR:
            with st.expander("Detalhes técnicos", expanded=False):
                st.code(_INFERENCE_IMPORT_ERROR, language="text")
        st.markdown(
            "Para habilitar o playground, instale as dependências no mesmo ambiente do Streamlit e reinicie o servidor."
        )
    elif not gnn_model_path.exists() or not gnn_dataset_path.exists():
        expected_model = gnn_model_path if gnn_model_path_str else DEFAULT_GNN_MODEL_PATH
        expected_dataset = gnn_dataset_path if gnn_dataset_path_str else DEFAULT_PYG_DATASET_PATH
        st.info(
            "Treine o GNN para ativar o playground. Esperamos encontrar os arquivos "
            f"`{expected_model}` e `{expected_dataset}`."
        )
    else:
        artefacts = get_inference_artifacts(gnn_model_path, gnn_dataset_path)
        if artefacts is None:
            st.warning("Não foi possível carregar os artefatos treinados do GNN.")
        else:
            node_metadata: List[Dict[str, Any]] = artefacts.node_metadata
            if not node_metadata:
                st.info("O snapshot atual não possui metadados de nós para inferência.")
            else:
                options = [
                    {
                        "index": idx,
                        "label": meta.get("label", meta.get("id", str(idx))) or str(idx),
                        "type": meta.get("type", "?"),
                        "id": meta.get("id", str(idx)),
                    }
                    for idx, meta in enumerate(node_metadata)
                ]
                options.sort(key=lambda item: (item["label"], item["type"], item["id"]))
                display_candidates = [f"{item['label']} · {item['type']} ({item['id']})" for item in options]
                label_to_index = {display: item["index"] for display, item in zip(display_candidates, options)}

                if not display_candidates:
                    st.info("Nenhuma entidade disponível para inferência.")
                else:
                    default_source_idx = 0
                    source_choice = st.selectbox(
                        "Entidade fonte",
                        display_candidates,
                        index=default_source_idx,
                        key="gnn_source",
                    )
                    source_index = label_to_index[source_choice]

                    remaining_targets = [label for label in display_candidates if label != source_choice]
                    if remaining_targets:
                        target_choice = st.selectbox(
                            "Entidade alvo",
                            remaining_targets,
                            index=0,
                            key="gnn_target",
                        )
                        target_index = label_to_index[target_choice]
                    else:
                        target_choice = ""
                        target_index = source_index

                    if predict_link_probability is not None and remaining_targets:
                        probability = predict_link_probability(artefacts, source_index, target_index)
                        st.metric("Probabilidade estimada", f"{probability:.3f}")

                    top_k_limit = max(1, min(20, len(display_candidates) - 1))
                    top_k_value = st.slider(
                        "Top-K recomendações",
                        min_value=1,
                        max_value=top_k_limit,
                        value=min(5, top_k_limit),
                        key="gnn_top_k",
                    )

                    if recommend_targets is not None:
                        recommendations = recommend_targets(
                            artefacts,
                            source_index=source_index,
                            top_k=top_k_value,
                            exclude_existing=True,
                        )
                        if recommendations:
                            rec_df = pl.DataFrame(recommendations)
                            rec_display = (
                                rec_df.select(["label", "type", "id", "score"])
                                .rename({"label": "Entity", "type": "Type", "id": "ID", "score": "Probability"})
                                .with_columns(pl.col("Probability").round(4))
                            )
                            st.dataframe(rec_display.to_pandas(), use_container_width=True)
                        else:
                            st.write("Nenhuma recomendação disponível para esta entidade.")
                    else:
                        st.info("Funções de recomendação indisponíveis (dependências faltantes).")

    st.subheader("Graph Visualization")
    if graph_snapshot_path is None:
        st.info("Nenhum snapshot de grafo disponível. Execute o pipeline para gerar um arquivo em `artifacts/graph_snapshot.json`.")
    else:
        _render_graph(graph_snapshot_path)

    st.subheader("Raw Metrics JSON")
    st.json(metrics, expanded=False)

    footer = st.empty()
    footer.caption(
        f"Metrics source: {metrics_path} • Last refreshed: {datetime.utcnow().isoformat()}"
    )


if __name__ == "__main__":
    main()
