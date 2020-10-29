import efselabwrapper

from efselabwrapper.pipeline import run_processing_pipeline, run_annotation_pipeline

LONG_STRING = '''
Till en början stack smittspridningen ut bland 20–29-åringar, för att sedan nå alla åldersgrupper.

– Smittspridningen gäller inte bara specifika situationer, miljöer eller arbetsplatser. Här är det mer en allmänspridning i samhället, säger hon.

De regioner som har högst smittspridning per capita är Uppsala, Örebro, Jämtland-Härjedalen, Östergötland, Jönköping och Stockholm, enligt Folkhälsomyndighetens senaste lägesrapport. Även Skåne har sett en stor ökning av antalet fall
'''

def main():
    print("-- MAIN --")
    corpus = [LONG_STRING]
    processed = run_processing_pipeline(corpus)
    print("-- COMPLETE --")
    print(processed)

if __name__ == '__main__':
    main()

