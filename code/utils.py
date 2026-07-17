""" Utility classes and functions related to MACT (NAACL 2025). 

Copyright (c) 2025 Robert Bosch GmbH 


This program is free software: you can redistribute it and/or modify 

it under the terms of the GNU Affero General Public License as published 

by the Free Software Foundation, either version 3 of the License, or 

(at your option) any later version. 

This program is distributed in the hope that it will be useful, 

but WITHOUT ANY WARRANTY; without even the implied warranty of 

MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 

GNU Affero General Public License for more details. 

You should have received a copy of the GNU Affero General Public License 

along with this program.  If not, see <https://www.gnu.org/licenses/>. 

"""

import os
import joblib
import json
import pandas as pd
import random
import tiktoken
import re
# random.seed(42)


def clean_cell(cell, idx, header) -> str:
    if not isinstance(cell, str):
        cell = str(cell)
    cell = cell.replace("\\n", " ")
    cell = cell.replace("\n", " ")
    if cell.strip() == "" and header:
        cell = f"column {idx+1}"
    return cell


def check_header(header: list) -> list:
    # check wether header repete
    if not len(set(header)) == len(header):
        header_ = []
        for cell in header:
            if cell in header:
                cell = cell + f"_{random.randint(0,1000)}"
            header_.append(cell)
    return header


def table_linear(table, num_row):

    header = table[0]
    header = [clean_cell(cell, i, header=True)
              for i, cell in enumerate(header)]
    header = check_header(header)
    if not num_row:
        num_row = len(table) - 1
    selected_rows = table[1:num_row+1]
    output = ""
    output += "| " + " | ".join(header) + " |"
    output += "\n"
    for row in selected_rows:
        row = [clean_cell(cell, i, header=False) for i, cell in enumerate(row)]
        output += "| " + " | ".join(row) + " |"
        output += "\n"
    return output


def normalize_table_rows(table, diagnostics=None):
    """Normalize relational rows to the header width and report repairs."""
    diagnostics = diagnostics if diagnostics is not None else []
    if not table:
        return table

    header = list(table[0])
    expected_width = len(header)
    normalized = [header]
    for row_index, row in enumerate(table[1:], start=1):
        row = list(row)
        actual_width = len(row)
        if actual_width < expected_width:
            row.extend([""] * (expected_width - actual_width))
            diagnostics.append({
                "source_row_index": row_index,
                "actual_width": actual_width,
                "expected_width": expected_width,
                "action": "padded_right",
            })
        elif actual_width > expected_width:
            row = row[:expected_width]
            diagnostics.append({
                "source_row_index": row_index,
                "actual_width": actual_width,
                "expected_width": expected_width,
                "action": "truncated_right",
            })
        normalized.append(row)
    return normalized


def table2df(table, diagnostics=None):
    # transform table in list format into executable dataframe code
    table = normalize_table_rows(table, diagnostics=diagnostics)
    output = "import pandas as pd\n"
    if not table:
        return output + "df=pd.DataFrame()"
    header = table[0]
    header = [clean_cell(cell, i, header=True)
              for i, cell in enumerate(header)]
    header = check_header(header)
    rows = table[1:]
    rows_ = []
    for row in rows:
        row_ = []
        for cell in row:
            try:
                cell = int(cell)
            except:
                pass
            try:
                cell = float(cell)
            except:
                pass
            if isinstance(cell, str):
                cell = cell.replace("\\n", " ")
                cell = cell.replace("\n", " ")
            row_.append(cell)
        rows_.append(row_)

    output += f"data_rows={rows_!r}\n"
    output += f"data_columns={header!r}\n"
    output += "df=pd.DataFrame(data_rows, columns=data_columns)"
    return output


def summarize_react_trial(agents):
    correct = [a for a in agents if a.is_correct()]
    halted = [a for a in agents if a.is_halted()]
    incorrect = [a for a in agents if a.is_finished() and not a.is_correct()]
    return correct, incorrect, halted


def dfcode2str(dfcode):
    data = re.findall(r'\{.+?\}', dfcode)[0]
    data = eval(data)
    df = pd.DataFrame(data)
    rows = df.values.tolist()
    header = df.columns.tolist()
    rows.insert(0, header)
    table_string = table_linear(rows, num_row=50)
    return table_string


def parse_action(string):
    first_action = None
    for match in re.finditer(r'\b(Retrieve|Operate|Finish|Search|Calculate|Verify)\[', string):
        if first_action is None or match.start() < first_action.start():
            first_action = match

    if first_action is None:
        return None, None

    action_type = first_action.group(1)
    argument_start = first_action.end()
    depth = 1
    quote = None
    escaped = False

    for index in range(argument_start, len(string)):
        char = string[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            previous_char = string[index - 1] if index > argument_start else ""
            next_char = string[index + 1] if index + 1 < len(string) else ""
            if char == "'" and previous_char.isalnum() and next_char.isspace():
                continue
            quote = char
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return action_type, string[argument_start:index]

    return None, None


def extract_from_outputs(outputs, num_choices):
    try:
        extracted = re.findall(r'The best path is \d', outputs) + \
            re.findall(r'The best result is \d', outputs)
        if len(extracted) > 0:
            target_choice = int(re.findall(r'\d', extracted[0])[0])-1
        else:
            target_choice = int(random.sample(num_choices, k=1)[0])
        # make sure target choice is within the num_choices
        if not target_choice < num_choices:
            target_choice = int(random.sample(num_choices, k=1)[0])
    except:
        target_choice = 0
    return target_choice


def get_databench_table(table_dir, dataset, k=2):
    df = pd.read_parquet(
        f"{table_dir}/{dataset}/all.parquet", engine='pyarrow')
    df_path = f"{table_dir}/{dataset}/all.parquet"
    header = df.columns.tolist()
    val_dict = {h: df[h].tolist() for h in header}
    vals = [val_dict[h][:k] for h in header]
    vals = [[col[i] for col in vals] for i in range(k)]
    vals.insert(0, header)
    table = table_linear(vals, num_row=None)
    x = len(df) - 3
    table += f"...[remaining {x} rows unshown due to large table size]..."
    return table, vals,  df_path
