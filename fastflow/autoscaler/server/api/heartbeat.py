from json.decoder import JSONDecodeError

from fastflow.autoscaler.utils.server_util import loads_json, dumps_json, set_headers


def _update_state(handler, client_status):
    client_key = client_status.get("instance_id")
    if client_key is None:
        raise ValueError
    handler.fastflow_state_set(client_key, client_status)


def do_POST(handler):
    client_status = None
    try:
        client_status = loads_json(handler)
    except (TypeError, ValueError, JSONDecodeError):
        handler.send_response(400, "Bad Request")

    response_dict = {}
    if client_status is None:
        response_dict["message"] = "Cannot understand your json request."
    elif client_status.get("uuid_token") != handler.uuid_token:
        handler.send_response(403, "Forbidden")
        response_dict["message"] = "uuid token does not match."
    else:
        try:
            _update_state(handler, client_status)
            handler.send_response(200, "Success")
            response_dict["message"] = "Success"
        except ValueError:
            handler.send_response(400, "Bad Request")
            response_dict["message"] = "client type or address does not exist"

    response_json = dumps_json(response_dict)
    set_headers(handler, len(response_json))
    handler.wfile.write(response_json)


def route(handler):
    if handler.command == "POST":
        do_POST(handler)
    else:
        handler.send_response(400, "Bad Request")
        set_headers(handler, 0)
