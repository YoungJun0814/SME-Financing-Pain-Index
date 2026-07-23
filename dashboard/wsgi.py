try:
    from dashboard.app import server
except ModuleNotFoundError:
    from app import server


application = server
