# %%
import copy


def update_dict(original, update, allow_new_keys):
    original = copy.deepcopy(original)
    for key, value in update.items():
        if isinstance(value, dict):
            if isinstance(original, dict):
                if (key not in original) and (not allow_new_keys):
                    raise KeyError(f"Key {key} not found in original dict. Possible keys are {list(original.keys())}")
                original_subdict = original.get(key, {})
                original[key] = update_dict(original_subdict, update[key], allow_new_keys)
            elif isinstance(original, list):
                list_index = int(key)
                if key >= len(original):
                    raise KeyError(
                        f"List index {list_index} exceeds length of list. Maximum index is {len(original) - 1}."
                    )
                original[list_index] = update_dict(original[list_index], update[key], allow_new_keys)
        else:
            original[key] = value
    return original


if __name__ == "__main__":
    o = dict(a=[dict(b=1)])
    update = dict(a={0: dict(b=2)})

    new = update_dict(o, update, False)
