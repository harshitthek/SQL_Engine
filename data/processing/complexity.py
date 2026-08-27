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
    if len(table_un
# [WIP: AST parsing traversal]
