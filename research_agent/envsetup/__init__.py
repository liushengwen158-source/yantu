"""科研入门 agent 的「环境配置」能力模块。

对外主要入口：
    from research_agent.envsetup import detect, load_knowledge, build_guidance
"""

from .advisor import build_guidance, guidance_to_text, render_actions, render_markdown
from .detector import detect, load_knowledge
from .installer import FixAction, build_plan, execute
from .models import CheckResult, EnvReport, Status

__all__ = [
    "detect",
    "load_knowledge",
    "build_guidance",
    "guidance_to_text",
    "render_markdown",
    "render_actions",
    "build_plan",
    "execute",
    "FixAction",
    "CheckResult",
    "EnvReport",
    "Status",
]
