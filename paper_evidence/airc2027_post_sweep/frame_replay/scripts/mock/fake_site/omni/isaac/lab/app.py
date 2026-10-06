"""FAKE AppLauncher (offline test harness). Not Isaac."""


class _FakeSimApp:
    def is_running(self):
        return True

    def close(self):
        pass


class AppLauncher:
    def __init__(self, args):
        self.app = _FakeSimApp()
        import os
        if os.environ.get("FAKE_KIT_PIL_MIX") == "1":
            # Reproduce what Isaac Sim 4.1 does once Kit starts (seen in logs/validate.log): PIL submodules
            # imported later come from Kit's prebundled Pillow 10.2.0 while PIL.Image stays 12.3.0.
            import PIL
            PIL.__path__.insert(0, os.path.expanduser(
                "~/miniconda3/envs/vlnce-isaac/lib/python3.10/site-packages/isaacsim/extscache/"
                "omni.kit.pip_archive/pip_prebundle/PIL"))

    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--livestream", type=int, default=-1)
        parser.add_argument("--enable_cameras", action="store_true", default=False)
        parser.add_argument("--device_id", type=int, default=0)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--experience", type=str, default="")
