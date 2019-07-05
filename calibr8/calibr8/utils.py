def to_did(parameter_name:str, replicate_id:str):
    if replicate_id is None:
        return parameter_name
    else:
        return "{}.{}".format(parameter_name, replicate_id)
