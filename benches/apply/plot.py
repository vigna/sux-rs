import matplotlib.pyplot as plt
import pandas as pd
import re

# We load up the benchmark file with the Criterion benchmark results
benchmark = open("bench_apply.log", "r").read()

# We split the file into lines and keep only the lines containing 'time:'
lines = benchmark.split("\n")

# We create a list of dictionaries, each dictionary containing the benchmark results for one function
results = []

uom = {
    "ps": 1e-12,
    "ns": 1e-9,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1,
}
uom_str = "("+"|".join(uom.keys())+")"

r = re.compile(r"(?P<word>u\d+)/(?P<name>\S+)/(?P<window>\d+)\s+time:\s+\[(?P<lower>\d+\.\d+)\s+(?P<lower_uom>{uom})\s+(?P<average>\d+\.\d+)\s+(?P<average_uom>{uom})\s+(?P<upper>\d+\.\d+)\s+(?P<upper_uom>{uom})\]".format(uom=uom_str))

for line in lines:
    if "time:" not in line:
        continue
    
    match = next(m.groupdict() for m in r.finditer(line))

    # We add the results to the list
    results.append({
        "name": match["name"],
        "word": match["word"],
        "window": int(match["window"]),
        "lower"  : float(match["lower"]) * uom[match["lower_uom"]] * 1e6,
        "average": float(match["average"]) * uom[match["average_uom"]] * 1e6,
        "upper"  : float(match["upper"]) * uom[match["upper_uom"]] * 1e6,
    })

# We create a DataFrame from the list of dictionaries
df = pd.DataFrame(results)

# We create a new dataframe with the name of the function combined with the time
# columns so to better display and compare the results in a human readable way
apply_df = df[df["name"] == "apply"].drop(columns=["name"])
luca_df = df[df["name"] == "luca"].drop(columns=["name"])
get_set_df = df[df["name"] == "get/set"].drop(columns=["name"])

apply_df.columns = ["word", "window", "lower_apply", "average_apply", "upper_apply"]
luca_df.columns = ["word", "window", "lower_apply", "average_apply", "upper_apply"]
get_set_df.columns = ["word", "window", "lower_get_set", "average_get_set", "upper_get_set"]
# We sort both dataframes by word and window
apply_df = apply_df.sort_values(by=["word", "window"])
luca_df = apply_df.sort_values(by=["word", "window"])
get_set_df = get_set_df.sort_values(by=["word", "window"])
# We drop the word and window columns from the get_set_df
get_set_df = get_set_df.drop(columns=["word", "window"])

human_df = pd.concat([
    apply_df.reset_index(drop=True),
    luca_df.reset_index(drop=True),
    get_set_df.reset_index(drop=True)
], axis=1)

human_df.to_csv("bench_apply.csv", index=False)

human_df


fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)

# We make vertical lines at 8, 16, 32 and 64 to highlight the window sizes
# matching with the word sizes
ax.axvline(8, color="black", linestyle=":", alpha=0.5)
ax.axvline(16, color="black", linestyle=":", alpha=0.5)
ax.axvline(32, color="black", linestyle=":", alpha=0.5)
ax.axvline(64, color="black", linestyle=":", alpha=0.5)
# We display the size of the word sizes on the x-axis
ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
ax.set_xticklabels(["4", "u8", "12", "u16", "20", "24", "28", "u32", "36", "40", "44", "48", "52", "56", "60", "u64"])

for name in df.name.unique():
    for word in reversed(sorted(df.word.unique())):
        
        subdf = df[(df.name == name) & (df.word == word)]
        # We sort the values by the window size
        subdf = subdf.sort_values("window")

        # We color the area above and below the average time
        area = ax.fill_between(subdf.window, y1=subdf.lower, y2=subdf.upper, alpha=1)

        # We plot the average time using the same color as per the area
        ax.plot(
            subdf.window,
            subdf.average,
            label=f"{name} {word}",
            color=area.get_facecolor()[0],
            alpha=1.0,
            linestyle="-",
        )

ax.set_xlabel("Window size")
ax.legend(ncol=2)
ax.set_ylabel("Time (µs)")
ax.set_title("Absolute time")
fig.tight_layout()
# We save the plot to a file
plt.savefig("bench_apply.jpg")

fig, ax = plt.subplots(1, 1, figsize=(14, 8), dpi=150)

# We make vertical lines at 8, 16, 32 and 64 to highlight the window sizes
# matching with the word sizes
ax.axvline(8, color="black", linestyle=":", alpha=0.5)
ax.axvline(16, color="black", linestyle=":", alpha=0.5)
ax.axvline(32, color="black", linestyle=":", alpha=0.5)
ax.axvline(64, color="black", linestyle=":", alpha=0.5)
# We display the size of the word sizes on the x-axis
ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])
ax.set_xticklabels(["4", "u8", "12", "u16", "20", "24", "28", "u32", "36", "40", "44", "48", "52", "56", "60", "u64"])

for name in df.name.unique():
    for word in reversed(sorted(df.word.unique())):
        
        subdf = df[(df.name == name) & (df.word == word)]
        # We sort the values by the window size
        subdf = subdf.sort_values("window")

        # We color the area above and below the average time
        area = ax.fill_between(subdf.window, y1=subdf.lower/ 1_000, y2=subdf.upper/ 1_000, alpha=1)

        # We plot the average time using the same color as per the area
        ax.plot(
            subdf.window,
            subdf.average/ 1_000,
            label=f"{name} {word}",
            color=area.get_facecolor()[0],
            alpha=1.0,
            linestyle="-",
        )

ax.set_xlabel("Window size")
ax.legend(ncol=2)
ax.set_ylabel("ns / element")
fig.tight_layout()
# We save the plot to a file
plt.savefig("bench_apply_per_element.jpg")