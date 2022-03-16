class Review():
    def __init__(self, date, writer, review, rating, type):
        self.date = date
        self.writer = writer
        self.review = review
        self.rating = rating
        self.type = type
    
    def __str__(self):
        s = ''
        s += self.date + '|'
        s += self.writer + '|'
        s += self.review + '|'
        s += self.rating + '|'
        s += self.type
        return s