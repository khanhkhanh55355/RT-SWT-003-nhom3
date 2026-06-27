import importlib
pkgs = ['pandas','numpy','scipy','matplotlib','seaborn']
for p in pkgs:
    try:
        importlib.import_module(p)
        print(p + ' OK')
    except Exception as e:
        print(p + ' FAILED: ' + str(e))
