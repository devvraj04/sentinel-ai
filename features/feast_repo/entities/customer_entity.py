from feast import Entity, ValueType
 
customer = Entity(
    name="customer_id",
    description="Unique customer identifier across all products",
    value_type=ValueType.STRING,
    tags={"owner": "sentinel-team"},
)
