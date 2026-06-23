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


def table2df(table):
    # transform table in list format into df code
    # currently only relational table
    output = "import pandas as pd\n"
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

    transposed_rows = [[] for i in range(len(header))]
    for line in rows_:
        for i, cell in enumerate(line):
            transposed_rows[i].append(cell)
    output += "data={"
    for h, v_row in zip(header, transposed_rows):
        output += f"'{h}':{v_row}"
        output += ","
    output = output[:-1]
    output += "}"
    output += "\n"
    output += "df=pd.DataFrame(data)"
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
    first_action = re.search(
        r'(Retrieve|Operate|Finish|Search|Calculate|Verify)\[.+?\]', string)
    if first_action and first_action.group(0).startswith("Verify["):
        actions = [first_action.group(0)]
    else:
        actions = re.findall(r'Retrieve\[.+?\]', string)+re.findall(r'Operate\[.+?\]', string)+re.findall(
            r'Finish\[.+?\]', string)+re.findall(r'Search\[.+?\]', string)+re.findall(r'Calculate\[.+?\]', string)+re.findall(r'Verify\[.+?\]', string)
    if actions:
        action = actions[0]
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, action)
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return None, None
    else:
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
