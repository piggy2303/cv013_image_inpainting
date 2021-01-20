from gevent.pywsgi import WSGIServer
from index_api import app

if __name__ == '__main__':
    print("done")
    http_server = WSGIServer(('0.0.0.0', 5017), app)
    http_server.serve_forever()
