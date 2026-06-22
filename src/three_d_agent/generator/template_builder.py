from pathlib import Path
from importlib import import_module
from three_d_agent.sad.schema import SAD


class TemplateBuilder:
    def supports(self, sad: SAD) -> tuple[bool, str]:
        try:
            import_module(f"three_d_agent.generator.templates.{sad.category}")
        except ModuleNotFoundError:
            return False, f"no template for category '{sad.category}'"
        return True, ""

    def build(self, sad: SAD, work_dir: Path) -> Path:
        module = import_module(f"three_d_agent.generator.templates.{sad.category}")
        return module.build(sad, work_dir)
