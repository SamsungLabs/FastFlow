from fastflow.autoscaler.utils.server_util import dumps_json, set_headers


def do_GET(handler):
    handler.send_response(200)
    response_json = dumps_json(handler.fastflow_state_get_all())
    set_headers(handler, len(response_json))
    handler.wfile.write(response_json)


def route(handler):
    if handler.command == "GET":
        do_GET(handler)
    else:
        handler.send_response(400, "Bad Request")
        set_headers(handler, 0)
