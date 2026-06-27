# Marks cc_lib as a package so `import cc_lib.<module>` is unambiguous.
# Kept empty on purpose: the widget hot-reloads by purging every cc_lib.*
# module from sys.modules (see purge_by_path in the widget) and re-importing,
# so nothing here should hold state that needs to survive a reload.
