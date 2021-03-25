class Subject_extractor:
    def __init__(self,verbose=False):
        self.verbose=verbose

    def extract_subject(self,request):
        return request.split(" ")[-1]

