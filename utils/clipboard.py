"""
클립보드 복사 버튼 유틸리티.
UTF-8 base64 인코딩으로 한글 텍스트도 안전하게 복사한다.
navigator.clipboard API → execCommand 순으로 폴백.
"""
from __future__ import annotations

import base64

import streamlit.components.v1 as components

_DEFAULT_GRADIENT = "linear-gradient(135deg,#7C3AED 0%,#5B21B6 100%)"


def copy_button(
    text: str,
    label: str = "📋 복사",
    gradient: str = _DEFAULT_GRADIENT,
    height: int = 55,
) -> None:
    """
    클립보드 복사 버튼을 HTML iframe으로 렌더링한다.

    Parameters
    ----------
    text:     복사할 내용 (UTF-8)
    label:    버튼에 표시할 텍스트
    gradient: CSS background 속성값 (기본: 보라색 그라디언트)
    height:   iframe 높이 (px)
    """
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    done_label = "✅ 복사 완료 — Claude.ai에 붙여넣으세요"
    html = f"""
<meta charset="utf-8">
<style>
  body{{margin:0;padding:0}}
  .btn{{background:{gradient};color:#fff;border:none;
        padding:13px 0;border-radius:10px;font-size:16px;font-weight:700;
        cursor:pointer;width:100%;transition:opacity .15s;letter-spacing:.3px}}
  .btn:hover{{opacity:.88}}.btn:active{{opacity:.72;transform:scale(.99)}}
</style>
<button class="btn" onclick="(function(btn){{
  var bytes=Uint8Array.from(atob('{b64}'),function(c){{return c.charCodeAt(0)}});
  var text=new TextDecoder('utf-8').decode(bytes);
  function done(){{
    btn.textContent='{done_label}';
    setTimeout(function(){{btn.textContent='{label}'}},4000)
  }}
  function fallback(t){{
    var doc=(window.parent&&window.parent!==window&&window.parent.document)
      ?window.parent.document:document;
    var ta=doc.createElement('textarea');ta.value=t;
    ta.style.cssText='position:fixed;opacity:0;top:0;left:0';
    doc.body.appendChild(ta);ta.focus();ta.select();
    try{{doc.execCommand('copy')}}catch(e){{}}
    doc.body.removeChild(ta);done();
  }}
  var nav=(window.parent&&window.parent!==window&&window.parent.navigator)
    ?window.parent.navigator:navigator;
  if(nav.clipboard&&nav.clipboard.writeText){{
    nav.clipboard.writeText(text).then(done).catch(function(){{fallback(text)}})
  }}else{{fallback(text)}}
}})(this)">{label}</button>"""
    components.html(html, height=height)
