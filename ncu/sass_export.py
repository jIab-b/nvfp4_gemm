#!/usr/bin/env python3
import argparse
import csv
import pathlib
import re


def extract_opcode(instr: str) -> str:
    """
    Return the opcode mnemonic used for de-duplication.
    Strips leading predicates (e.g. '@P0') and grabs the first token.
    """
    text = instr.strip()
    text = re.sub(r"^@\S+\s+", "", text)
    parts = text.split()
    return parts[0] if parts else ""


def filter_commands(input_path: pathlib.Path, output_path: pathlib.Path) -> int:
    seen = set()
    kept_rows = []

    with input_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            # Skip header if present.
            if row[0] == "Kernel Name" or row[1] == "Source":
                continue

            opcode = extract_opcode(row[1])
            if not opcode or opcode in seen:
                continue

            seen.add(opcode)
            kept_rows.append(row)

    if not kept_rows:
        return 0

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(kept_rows)

    return len(kept_rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export first occurrence of each SASS opcode from an Nsight CSV."
    )
    parser.add_argument("input", help="Input CSV file (e.g. sub_tc2_sass.csv)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path (default: <input_stem>_commands.csv)",
    )

    args = parser.parse_args(argv)
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_commands.csv")

    count = filter_commands(input_path, output_path)
    print(f"Wrote {count} unique commands to {output_path}")


if __name__ == "__main__":
    main()
