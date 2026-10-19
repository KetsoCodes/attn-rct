"""Fill the queue with the next Text runs, up to the cluster's submit limit.

bigbatch allows 48 submitted jobs per user (pending + running, each array task counted),
and the Text grid is 120 runs, so it has to go in waves. Run this whenever there is room;
it works out what is still needed and submits only that. Safe to run as often as you like:
finished runs (a result file exists) and runs already in the queue are never resubmitted.

Order matters. Analysis plan Amendment 02 pre-registers a fallback: if the full grid cannot
finish in time, Text uses the balanced half-fraction of the design space, designs d001,
d002, d004, d007. Submitting those four designs FIRST means that if time runs short, the
fallback set is already complete -- the pre-registered decision then needs no judgement
made after seeing results.

    python cluster/submit_text.py            # show what would be submitted
    python cluster/submit_text.py --submit   # submit it
"""

from __future__ import annotations

import argparse
import csv
import getpass
import pathlib
import subprocess

SUBMIT_LIMIT = 48
CONCURRENT = 6
# Half-fraction (I = +ABC) first, then the remaining four designs.
DESIGN_ORDER = [1, 2, 4, 7, 0, 3, 5, 6]


def queued_jobs():
    """Every job of ours in the queue, one entry per array task, as SLURM ids."""
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-r", "-o", "%i"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return out


def queued_rows(job_ids):
    """Manifest rows already queued under the grid's job name, from 'JOB_TASK' ids."""
    out = subprocess.run(
        ["squeue", "-u", getpass.getuser(), "-h", "-r", "--name=attnrct", "-o", "%i"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    rows = set()
    for job in out:
        if "_" in job:
            task = job.split("_", 1)[1]
            if task.isdigit():
                rows.add(int(task))
    return rows


def compress(rows):
    """[361, 362, 363, 370] -> '361-363,370', Slurm's array syntax."""
    parts, start, prev = [], None, None
    for r in rows:
        if start is None:
            start = prev = r
        elif r == prev + 1:
            prev = r
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = r
    if start is not None:
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="cluster/runs.csv")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    text = [(i, r) for i, r in enumerate(rows, start=1) if r["task"] == "text"]
    if not text:
        raise SystemExit("no Text rows in the manifest -- run cluster/make_manifest.py")

    in_queue = queued_rows(None)
    done, needed = [], []
    for design in DESIGN_ORDER:
        for i, r in text:
            if int(r["design_id"]) != design:
                continue
            if (pathlib.Path(args.results_dir) / f"{r['run_id']}.json").exists():
                done.append(i)
            elif i not in in_queue:
                needed.append(i)

    room = SUBMIT_LIMIT - len(queued_jobs())
    batch = sorted(needed[:max(room, 0)])
    print(f"Text runs: {len(done)}/120 done, {len(in_queue)} in queue, "
          f"{len(needed)} still to submit")
    print(f"queue room: {room} of {SUBMIT_LIMIT}")

    if not batch:
        print("nothing to submit now" + (" -- all Text runs are done or queued."
                                         if not needed else " -- the queue is full; "
                                         "run this again later."))
        return

    command = ["sbatch", f"--array={compress(batch)}%{CONCURRENT}",
               "cluster/train_array.slurm"]
    print("next batch:", " ".join(command))
    if args.submit:
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout.strip() or result.stderr.strip())
    else:
        print("(dry run -- add --submit to send it)")


if __name__ == "__main__":
    main()
