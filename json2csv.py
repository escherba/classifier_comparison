import csv
import json
import sys
from argparse import ArgumentParser, FileType

p = ArgumentParser()
p.add_argument("--fields", type=str, default=[], nargs='*', required=True,
               help="Field to select")
p.add_argument("--input", type=FileType('r'), default=sys.stdin)
p.add_argument("--output", type=FileType('w'), default=sys.stdout)

args = p.parse_args()

writer = csv.writer(args.output,
                    delimiter=",",
                    quotechar='"',
                    quoting=csv.QUOTE_NONNUMERIC)

for line in args.input:
    try:
        o = json.loads(line)
    except ValueError:
        continue
    fields = [o[f] for f in args.fields]
    writer.writerow(fields)

args.input.close()
args.output.close()
