import os

IS_RUNNING_WITH_STREAMLIT = os.environ.get("STREAMLIT_RUNNING") == "true"

_CACHE = {}  # Simple cache outside Streamlit

if IS_RUNNING_WITH_STREAMLIT:
    import streamlit as st
    @st.cache_data
    def get_or_update_cache(_new_cache=None):
        if _new_cache is None:
            return {}
        return _new_cache

def update_cache(**updates):
    if IS_RUNNING_WITH_STREAMLIT:
        new_cache = dict(get_or_update_cache(), **updates)
        get_or_update_cache.clear()
        return get_or_update_cache(new_cache)
    else:
        global _CACHE
        _CACHE.update(updates)
        return _CACHE

def get_cache(key):
    if IS_RUNNING_WITH_STREAMLIT:
        return get_or_update_cache().get(key)
    else:
        return _CACHE[key]
    
def has_cache(key):
    if IS_RUNNING_WITH_STREAMLIT:
        return key in get_or_update_cache()
    else:
        return key in _CACHE

def clear_cache():
    if IS_RUNNING_WITH_STREAMLIT:
        get_or_update_cache.clear()
    else:
        global _CACHE
        _CACHE = {}