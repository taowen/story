import os
from typing import Any, Dict

IS_RUNNING_WITH_STREAMLIT = os.environ.get("STREAMLIT_RUNNING") == "true"

_STATE: Dict[str, Any] = {}  # Simple cache outside Streamlit

if IS_RUNNING_WITH_STREAMLIT:
    import streamlit as st
    @st.cache_data
    def get_or_update_key(_new_cache: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if _new_cache is None:
            return {}
        return _new_cache

def update_key(**updates: Any) -> Dict[str, Any]:
    if IS_RUNNING_WITH_STREAMLIT:
        new_cache = dict(get_or_update_key(), **updates)
        get_or_update_key.clear()
        return get_or_update_key(new_cache)
    else:
        global _STATE
        _STATE.update(updates)
        return _STATE

def get_key(key: str) -> Any:
    if IS_RUNNING_WITH_STREAMLIT:
        return get_or_update_key().get(key)
    else:
        return _STATE[key]
    
def has_key(key: str) -> bool:
    if IS_RUNNING_WITH_STREAMLIT:
        return key in get_or_update_key()
    else:
        return key in _STATE