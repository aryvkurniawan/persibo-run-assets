"""Match analytics (post-processing on tracks + roles + field positions).

Public surface:
    Phase 6 ball possession — compute_possession, frame_possessor,
        smooth_possession, PossessionResult, export_possession,
        draw_possession_bar
    Phase 5 speed & distance — compute_analytics, compute_track_analytics,
        AnalyticsResult, TrackAnalytics, export_analytics,
        build_speed_lookup, draw_speed_labels
    Phase 6 player performance — compute_player_analytics, PlayerPerformance,
        export_player_analytics, generate_insight, player_rating
"""

from analytics.analytics_exporter import (
    export_analytics,
    export_analytics_txt,
    export_player_analytics,
    export_player_analytics_txt,
)
from analytics.analytics_models import AnalyticsResult, TrackAnalytics
from analytics.analytics_visualizer import (
    build_speed_lookup,
    build_stat_lookup,
    draw_player_stat_bars,
    draw_speed_labels,
)
from analytics.heatmap_generator import collect_positions, density_grid
from analytics.heatmap_renderer import render_heatmap, save_heatmap
from analytics.match_report import (
    MatchReport,
    build_match_report,
    export_final_report,
    export_final_report_txt,
)
from analytics.player_analytics import (
    PlayerPerformance,
    compute_player_analytics,
    compute_player_performance,
)
from analytics.possession import (
    PossessionResult,
    compute_possession,
    draw_possession_bar,
    export_possession,
    export_possession_txt,
    frame_possessor,
    smooth_possession,
)
from analytics.possession_engine import (
    PossessionAnalysis,
    compute_field_possession,
    export_field_possession,
    export_field_possession_txt,
    frame_owner,
)
from analytics.rating_engine import generate_insight, player_rating
from analytics.speed_distance import compute_analytics, compute_track_analytics
from analytics.tactical_analysis import (
    TacticalProfile,
    compute_tactical_analysis,
    compute_team_tactics,
    export_team_tactical,
    export_team_tactical_txt,
)

__all__ = [
    "compute_possession",
    "frame_possessor",
    "smooth_possession",
    "PossessionResult",
    "export_possession",
    "export_possession_txt",
    "draw_possession_bar",
    "compute_analytics",
    "compute_track_analytics",
    "AnalyticsResult",
    "TrackAnalytics",
    "export_analytics",
    "export_analytics_txt",
    "build_speed_lookup",
    "build_stat_lookup",
    "draw_speed_labels",
    "draw_player_stat_bars",
    "compute_player_analytics",
    "compute_player_performance",
    "PlayerPerformance",
    "export_player_analytics",
    "export_player_analytics_txt",
    "generate_insight",
    "player_rating",
    "collect_positions",
    "density_grid",
    "render_heatmap",
    "save_heatmap",
    "compute_tactical_analysis",
    "compute_team_tactics",
    "TacticalProfile",
    "export_team_tactical",
    "export_team_tactical_txt",
    "compute_field_possession",
    "frame_owner",
    "PossessionAnalysis",
    "export_field_possession",
    "export_field_possession_txt",
    "build_match_report",
    "MatchReport",
    "export_final_report",
    "export_final_report_txt",
]
