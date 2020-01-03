class TicketIndex(object):
    def __init__(self, name, tag, desc):
        self.name = name
        self.tag = tag
        self.desc: Desc = desc

class Desc(object):
    def __init__(self, underlyingCause, problemDetails):
        self.underlyingCause = underlyingCause
        self.problemDetails = problemDetails