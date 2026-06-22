import sys
from typing import Callable


def auto_accept_callback(version_summary: str) -> str:
    """Non-interactive callback: print the summary and always accept."""
    print(version_summary)
    print("> accept (auto)")
    return "accept"


def make_interactive_callback(
    in_stream=None, out_stream=None
) -> Callable[[str], str]:
    """Build a human callback that prints each version summary and reads one
    line of natural-language feedback from stdin.

    An empty line (or EOF / closed stdin) is treated as "accept", so the loop
    terminates cleanly when piped or run non-interactively.
    """
    istream = in_stream if in_stream is not None else sys.stdin
    ostream = out_stream if out_stream is not None else sys.stdout

    def callback(version_summary: str) -> str:
        print(version_summary, file=ostream)
        print("> (enter feedback, or blank to accept)", file=ostream)
        ostream.flush()
        line = istream.readline()
        if not line:
            return "accept"
        reply = line.strip()
        return reply or "accept"

    return callback
