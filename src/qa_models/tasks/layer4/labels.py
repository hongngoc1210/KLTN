"""Fixed four-action taxonomy, independent of diagnosis labels."""

REPAIR_ACTIONS = ("replace_span", "delete_span", "delete_claim", "rewrite_claim")
LABEL_TO_ID = {label: index for index, label in enumerate(REPAIR_ACTIONS)}
ID_TO_LABEL = dict(enumerate(REPAIR_ACTIONS))
