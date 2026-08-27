"""SQL Query Complexity Classifier based on Spider official evaluation benchmark.

Classifies SQL queries into four complexity tiers:
- simple (easy)
- medium
- hard
- extra-hard (extra)

Complexity is evaluated based on:
1. Component 1: Structural syntax (WHERE, GROUP BY, ORDER BY, LIMIT, JOINs, OR conditions, LIKE ops)
2. Component 2: Nested subqueries (nested in WHERE/HAVING/FROM, INTERSECT, UNION, EXCEPT)
3. Others: Aggregation counts, number of selected columns, number of WHERE conditions, number of GROUP BY columns
"""

from typing import Any, Dict, List, Tuple

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def has_agg(unit: Tuple[Any, ...]) -> bool:
    """Check whether a column or expression unit has an aggregate function."""
    return unit[0] != AGG_OPS.index("none")


def count_agg(units: List[Any]) -> int:
    """Count number of aggregate functions in the given units."""
    return len([unit for unit in units if has_agg(unit)])


def get_nested_sql(sql: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recursively extract nested SQL queries from conditions and set operations."""
    nested = []
    cond_units = (
        sql.get("from", {}).get("conds", [])[::2]
        + sql.get("where", [])[::2]
        + sql.get("having", [])[::2]
    )
    for cond_unit in cond_units:
        if len(cond_unit) > 3 and isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if len(cond_unit) > 4 and isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    if sql.get("intersect") is not None:
        nested.append(sql["intersect"])
    if sql.get("except") is not None:
        nested.append(sql["except"])
    if sql.get("union") is not None:
        nested.append(sql["union"])
    return nested


def count_component1(sql: Dict[str, Any]) -> int:
    """Count structural components: WHERE, GROUP BY, ORDER BY, LIMIT, JOINs, OR, LIKE."""
    count = 0
    if len(sql.get("where", [])) > 0:
        count += 1
    if len(sql.get("groupBy", [])) > 0:
        count += 1
    if len(sql.get("orderBy", [])) > 0:
        count += 1
    if sql.get("limit") is not None:
        count += 1
    table_units = sql.get("from", {}).get("table_units", [])
    if len(table_units) > 0:  # JOIN count = number of tables - 1
        count += len(table_units) - 1

    ao = (
        sql.get("from", {}).get("conds", [])[1::2]
        + sql.get("where", [])[1::2]
        + sql.get("having", [])[1::2]
    )
    count += len([token for token in ao if token == "or"])

    cond_units = (
        sql.get("from", {}).get("conds", [])[::2]
        + sql.get("where", [])[::2]
        + sql.get("having", [])[::2]
    )
    like_op_idx = WHERE_OPS.index("like")
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == like_op_idx])

    return count


def count_component2(sql: Dict[str, Any]) -> int:
    """Count nested SQL queries."""
    nested = get_nested_sql(sql)
    return len(nested)


def count_others(sql: Dict[str, Any]) -> int:
    """Count aggregate counts, multiple select columns, multiple conditions, multiple group bys."""
    count = 0
    select_units = sql.get("select", [False, []])[1]
    agg_count = count_agg(select_units)
    agg_count += count_agg(sql.get("where", [])[::2])
    agg_count += count_agg(sql.get("groupBy", []))
    if len(sql.get("orderBy", [])) > 0:
        order_units = sql["orderBy"][1]
        agg_count += count_agg(
            [unit[1] for unit in order_units if unit[1]]
            + [unit[2] for unit in order_units if unit[2]]
        )
    agg_count += count_agg(sql.get("having", []))
    if agg_count > 1:
        count += 1

    if len(select_units) > 1:
        count += 1

    if len(sql.get("where", [])) > 1:
        count += 1

    if len(sql.get("groupBy", [])) > 1:
        count += 1

    return count


def eval_hardness(sql: Dict[str, Any]) -> str:
    """Categorize SQL query into simple, medium, hard, extra-hard."""
    count_comp1_ = count_component1(sql)
    count_comp2_ = count_component2(sql)
    count_others_ = count_others(sql)

    if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
        return "simple"
    elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
        count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
    ):
        return "medium"
    elif (
        (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
        or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
        or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
    ):
        return "hard"
    else:
        return "extra-hard"
