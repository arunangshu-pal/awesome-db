use common::query::{
    Query, QueryOp, ProjectData, FilterData,
    CrossData, ScanData, Predicate,
    ComparisionOperator, ComparisionValue
};

fn main() {
    let query = Query {
        root: QueryOp::Project(ProjectData {
            column_name_map: vec![
                ("o_orderkey".to_string(), "o_orderkey".to_string()),
                ("c_name".to_string(), "c_name".to_string()),
            ],
            underlying: Box::new(QueryOp::Filter(FilterData {
                predicates: vec![
                    Predicate {
                        column_name: "o_custkey".to_string(),
                        operator: ComparisionOperator::EQ,
                        value: ComparisionValue::Column("c_custkey".to_string()),
                    }
                ],
                underlying: Box::new(QueryOp::Cross(CrossData {
                    left: Box::new(QueryOp::Scan(ScanData {
                        table_id: "orders".to_string(),
                    })),
                    right: Box::new(QueryOp::Scan(ScanData {
                        table_id: "customer".to_string(),
                    })),
                })),
            })),
        }),
    };

    let query_json = serde_json::to_string_pretty(&query).unwrap();

    println!("{}", query_json);
}
