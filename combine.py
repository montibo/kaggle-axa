def values_for_file(filename):
    values = {}
    with open(filename, "r") as trainfile:
        trainfile.readline()  # skip header
        for line in trainfile:
            items = line.split(",", 2)
            name = items[0]
            driver, trace = name.split("_", 2)
            if driver not in values.keys():
                values[driver] = {}
            values[driver][trace] = float(items[1])
    return values

if __name__ == '__main__':
    filename = "pyRegression_17_32_December_30_2014.csv"
    values1 = values_for_file(filename)
    filename = "pyRegression_21_06_December_30_2014.csv"
    values2 = values_for_file(filename)
    filename = "pyRegression_21_54_January_08_2015.csv"
    values3 = values_for_file(filename)

    for driver in values1.keys():
        for trace in values1[driver].keys():
            values1[driver][trace] = (1.0/3.0)*(values1[driver][trace] + values2[driver][trace] + values3[driver][trace])

    with open("pyRegression_combined.csv", 'w') as writefile:
        writefile.write("driver_trip,prob\n")
        for driver in values1.keys():
            for trace in values1[driver].keys():
                writefile.write("%s_%s, %.5f\n" % (driver, trace, values1[driver][trace]))
