import json
from pathlib import Path


DASHBOARD_PATH = (
    Path(__file__).parent
    / "grafana"
    / "dashboards"
    / "vllm.json"
)

DATASOURCE = {
    "type": "prometheus",
    "uid": "prometheus",
}


def target(expr: str, legend: str = ""):
    return {
        "expr": expr,
        "legendFormat": legend,
        "refId": "A",
    }


def stat_panel(
    panel_id: int,
    title: str,
    expr: str,
    x: int,
    y: int,
    unit: str = "short",
):
    return {
        "id": panel_id,
        "type": "stat",
        "title": title,
        "datasource": DATASOURCE,
        "gridPos": {
            "x": x,
            "y": y,
            "w": 4,
            "h": 4,
        },
        "targets": [
            target(expr),
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {
                "calcs": ["lastNotNull"],
                "fields": "",
                "values": False,
            },
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto",
        },
    }


def timeseries_panel(
    panel_id: int,
    title: str,
    targets: list,
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    unit: str = "short",
):
    return {
        "id": panel_id,
        "type": "timeseries",
        "title": title,
        "datasource": DATASOURCE,
        "gridPos": {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        },
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "unit": unit,
            },
            "overrides": [],
        },
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom",
            }
        },
    }


def build_dashboard():
    panels = [
        # -----------------------------------------------------
        # Ray Serve
        # -----------------------------------------------------

        stat_panel(
            1,
            "HTTP ongoing",
            "sum(ray_serve_num_ongoing_http_requests)",
            0,
            0,
        ),

        stat_panel(
            2,
            "Serve processing",
            "sum(ray_serve_replica_processing_queries)",
            4,
            0,
        ),

        stat_panel(
            3,
            "Serve queued",
            "sum(ray_serve_deployment_queued_queries)",
            8,
            0,
        ),

        # -----------------------------------------------------
        # vLLM scheduler
        # -----------------------------------------------------

        stat_panel(
            4,
            "vLLM running",
            "sum(ray_vllm_num_requests_running)",
            12,
            0,
        ),

        stat_panel(
            5,
            "vLLM waiting",
            "sum(ray_vllm_num_requests_waiting)",
            16,
            0,
        ),

        stat_panel(
            6,
            "KV cache",
            "100 * avg(ray_vllm_kv_cache_usage_perc)",
            20,
            0,
            unit="percent",
        ),

        # -----------------------------------------------------
        # Throughput / speculative decoding
        # -----------------------------------------------------

        stat_panel(
            7,
            "Output tokens/s",
            "sum(rate(ray_vllm_generation_tokens_total[1m]))",
            0,
            4,
            unit="ops",
        ),

        stat_panel(
            8,
            "Input tokens/s",
            "sum(rate(ray_vllm_prompt_tokens_total[1m]))",
            4,
            4,
            unit="ops",
        ),

        stat_panel(
            9,
            "Requests/s",
            "sum(rate(ray_serve_deployment_request_counter_total[1m]))",
            8,
            4,
            unit="reqps",
        ),

        stat_panel(
            10,
            "MTP acceptance",
            """
            100 *
            sum(rate(ray_vllm_spec_decode_num_accepted_tokens_total[1m]))
            /
            sum(rate(ray_vllm_spec_decode_num_draft_tokens_total[1m]))
            """,
            12,
            4,
            unit="percent",
        ),

        stat_panel(
            11,
            "Prompt cache hit",
            """
            100 *
            sum(rate(ray_vllm_prompt_tokens_cached_total[5m]))
            /
            sum(rate(ray_vllm_prompt_tokens_total[5m]))
            """,
            16,
            4,
            unit="percent",
        ),

        # -----------------------------------------------------
        # Scheduler over time
        # -----------------------------------------------------

        timeseries_panel(
            20,
            "vLLM scheduler",
            [
                target(
                    "sum(ray_vllm_num_requests_running)",
                    "running",
                ),
                {
                    "expr": "sum(ray_vllm_num_requests_waiting)",
                    "legendFormat": "waiting",
                    "refId": "B",
                },
            ],
            0,
            8,
        ),

        timeseries_panel(
            21,
            "Token throughput",
            [
                target(
                    "sum(rate(ray_vllm_generation_tokens_total[1m]))",
                    "output tokens/s",
                ),
                {
                    "expr": "sum(rate(ray_vllm_prompt_tokens_total[1m]))",
                    "legendFormat": "input tokens/s",
                    "refId": "B",
                },
            ],
            12,
            8,
        ),

        # -----------------------------------------------------
        # Latencies
        # -----------------------------------------------------

        timeseries_panel(
            30,
            "vLLM latency",
            [
                target(
                    """
                    sum(rate(ray_vllm_request_prefill_time_seconds_sum[5m]))
                    /
                    sum(rate(ray_vllm_request_prefill_time_seconds_count[5m]))
                    """,
                    "prefill",
                ),
                {
                    "expr": """
                    sum(rate(ray_vllm_request_decode_time_seconds_sum[5m]))
                    /
                    sum(rate(ray_vllm_request_decode_time_seconds_count[5m]))
                    """,
                    "legendFormat": "decode",
                    "refId": "B",
                },
                {
                    "expr": """
                    sum(rate(ray_vllm_request_queue_time_seconds_sum[5m]))
                    /
                    sum(rate(ray_vllm_request_queue_time_seconds_count[5m]))
                    """,
                    "legendFormat": "queue",
                    "refId": "C",
                },
            ],
            0,
            16,
            unit="s",
        ),

        timeseries_panel(
            31,
            "End-to-end latency",
            [
                target(
                    """
                    sum(rate(ray_vllm_e2e_request_latency_seconds_sum[5m]))
                    /
                    sum(rate(ray_vllm_e2e_request_latency_seconds_count[5m]))
                    """,
                    "vLLM",
                ),
                {
                    "expr": """
                    (
                      sum(rate(ray_serve_http_request_latency_ms_sum[5m]))
                      /
                      sum(rate(ray_serve_http_request_latency_ms_count[5m]))
                    ) / 1000
                    """,
                    "legendFormat": "HTTP",
                    "refId": "B",
                },
            ],
            12,
            16,
            unit="s",
        ),
    ]

    return {
        "uid": "lellray-vllm",
        "title": "LeLRay - vLLM",
        "tags": ["ray", "vllm", "htr"],
        "timezone": "browser",
        "schemaVersion": 41,
        "version": 1,
        "refresh": "5s",
        "time": {
            "from": "now-15m",
            "to": "now",
        },
        "panels": panels,
    }


def generate_dashboard():
    dashboard = build_dashboard()

    DASHBOARD_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DASHBOARD_PATH.write_text(
        json.dumps(
            dashboard,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Grafana dashboard generated: {DASHBOARD_PATH}"
    )


if __name__ == "__main__":
    generate_dashboard()