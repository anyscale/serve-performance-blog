from locust import HttpUser, task, between, events

PAYLOAD = ""


@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--token", type=str, default="", help="Auth bearer token")
    parser.add_argument(
        "--route_prefix",
        type=str,
        default=f"/echo?message={PAYLOAD}",
        help="Route prefix",
    )


@events.test_start.add_listener
def _(environment, **kw):
    pass


class MyUser(HttpUser):
    wait_time = between(0, 0)

    def on_start(self):
        if self.environment.parsed_options.token:
            self.client.headers.update(
                {"Authorization": f"Bearer {self.environment.parsed_options.token}"}
            )

    @task
    def index(self):
        self.client.get(
            self.environment.parsed_options.route_prefix,
            headers={"x-request-disconnect-disabled": "?1"},
            name="/echo",
        )
