import efselabwrapper

from efselabwrapper import run_processing_pipeline, run_annotation_pipeline, run_processing_pipeline_ner

LONG_STRING = '''
Till en början stack smittspridningen ut bland 20–29-åringar, för att sedan nå alla åldersgrupper.

– Smittspridningen gäller inte bara specifika situationer, miljöer eller arbetsplatser. Här är det mer en allmänspridning i samhället, säger Erik Stenemo.

De regioner som har högst smittspridning per capita är Uppsala, Örebro, Jämtland-Härjedalen, Östergötland, Jönköping och Stockholm, enligt Folkhälsomyndighetens senaste lägesrapport. Även Skåne har sett en stor ökning av antalet fall
'''

def main():
    print("-- MAIN --")
    corpus = [LONG_STRING]
    processed, ner = run_processing_pipeline_ner(corpus)
    print("-- COMPLETE --")

if __name__ == '__main__':
    main()

