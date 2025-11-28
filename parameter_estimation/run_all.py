from e1 import main as e1_main, DEFAULT_OPTIONS as e1_options
from e2 import main as e2_main, DEFAULT_OPTIONS as e2_options
from e3 import main as e3_main, DEFAULT_OPTIONS as e3_options
from e4 import main as e4_main, DEFAULT_OPTIONS as e4_options
from e5 import main as e5_main, DEFAULT_OPTIONS as e5_options
from e6 import main as e6_main, DEFAULT_OPTIONS as e6_options
from e7 import main as e7_main, DEFAULT_OPTIONS as e7_options
from e8 import main as e8_main, DEFAULT_OPTIONS as e8_options
from e9 import main as e9_main, DEFAULT_OPTIONS as e9_options

# %%

cases = {
    "e1": {
        "main": e1_main,
        "options": e1_options,
    },
    "e2": {
        "main": e2_main,
        "options": e2_options,
    },
    "e3": {
        "main": e3_main,
        "options": e3_options,
    },
    "e4": {
        "main": e4_main,
        "options": e4_options,
    },
    "e5": {
        "main": e5_main,
        "options": e5_options,
    },
    "e6": {
        "main": e6_main,
        "options": e6_options,
    },
    # Uses E6 data to account for larger particle porosity
    "e8": {
        "main": e8_main,
        "options": e8_options,
    },
    # Re-fit particle porosity to account for larger proteins; this is accounted for when setting capacity
    "e7": {
        "main": e7_main,
        "options": [
            {**e7_options, "include_axial_dispersion": False, "include_film_diffusion": False},
            {**e7_options, "include_axial_dispersion": True, "include_film_diffusion": False},
            {**e7_options, "include_axial_dispersion": True, "include_film_diffusion": True},
        ],
    },
    # Continue with protein specific axial dispersion, and non-limiting film diffusion
    "e9": {
        "main": e9_main,
        "options": [
            {**e9_options, "pH": 4.0},
            {**e9_options, "pH": 4.5},
            {**e9_options, "pH": 5.0},
        ],
    },
}


if __name__ == "__main__":

    prior_branch_name = None
    debug = False

    # %%
    for name, case_info in cases.items():
        print(f"Case: {name}")
        main_func = case_info["main"]
        options = case_info["options"]
        if not isinstance(options, list):
            options = [options]

        # Iterate over options
        results = []
        branch_names = []
        for opt in options:
            # Update options
            options = {
                **opt,
                "prior_branch_name": prior_branch_name,
                "debug": debug,
            }
            print(f"Running with options: {options}")

            # Run optimization
            posterior_branch_name, optimization_result = main_func(options)
            results.append(optimization_result)
            branch_names.append(posterior_branch_name)

        # Select the branch_name to use for the next case
        if name == "e7":
            # Use the second combination's branch_name for E7
            prior_branch_name = branch_names[1]
        elif name == "e9":
            # Use the last branch_name for E9 (or any other logic)
            prior_branch_name = branch_names[-1]
        else:
            # For other cases, use the only branch_name
            prior_branch_name = branch_names[0]

