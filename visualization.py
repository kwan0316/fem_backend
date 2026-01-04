import io
import base64
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")


def generate_frame_diagrams(ss):
    diagrams = {}

    def _capture_plot(plot_func, key):
        try:
            plt.figure(figsize=(10, 6))
            plot_func()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            diagrams[key] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()
        except Exception:
            diagrams[key] = None
            plt.close()

    _capture_plot(ss.show_structure, "structure")
    _capture_plot(ss.show_shear_force, "shear")
    _capture_plot(ss.show_axial_force, "axial")
    _capture_plot(ss.show_bending_moment, "moment")
    _capture_plot(ss.show_displacement, "displacement")

    return diagrams


