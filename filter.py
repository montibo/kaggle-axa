import operator

filename = "pyRegression_21_06_December_30_2014.csv"
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


for driver in [str(dr) for dr in range (2620,2631)]:
    print "---\n{0}\n---".format(driver)
    sorted_values = sorted(values[driver].items(), key=operator.itemgetter(1), reverse=True)
    for name, value in sorted_values[0:10]:
        print "%s: %s" % (name, value)
    sorted_values = sorted(values[driver].items(), key=operator.itemgetter(1))
    for name, value in sorted_values[0:10]:
        print "%s: %s" % (name, value)
