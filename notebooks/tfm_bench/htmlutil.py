"""Make report pages self-contained: inline plotly.js instead of loading it from a CDN.

A CDN <script> fails wherever the page is opened without internet access or inside a sandboxed preview
(-> "Plotly is not defined"), so every page embeds bench/vendor/plotly-2.35.2.min.js (MIT licence, ~4.6 MB)."""
import os

CDN_TAG = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "plotly-2.35.2.min.js")


def inline_plotly(html: str) -> str:
    assert CDN_TAG in html, "template no longer loads plotly 2.35.2; update htmlutil.CDN_TAG"
    js = open(VENDOR, encoding="utf-8").read().replace("</script", "<\\/script")
    return html.replace(CDN_TAG, "<script>" + js + "</script>")
