import os

class pos:
    def __init__(self,x,y):
        self.x=x
        self.y=y

class action:
    def __init__(self, classes):
        self.action="No action"
        self.next_poste="No poste"
        self.classes=classes
        self.stop=0
        self.message=""
        # self.position=[]

    # def add_pos():
    #     self.position.append()

    def add_action(self, current_classe):
        # print("current_classe : ",current_classe)
        if current_classe==self.classes[0]: #stop
            self.message="Robot stopped !"
            self.stop=1

        if self.action=="Validation_step": #validation step
            if current_classe == self.classes[-1]:
                self.message=str("Let's go to the station " + self.next_poste+"!")
                self.action="Action"
            if current_classe == self.classes[-2]:
                self.message="Action canceled !"
                self.action="Action"

        if current_classe in self.classes[1:-2]:
            self.action="Validation_step"
            self.next_poste=current_classe
            self.message=str("Do you validate station "+current_classe+ "?")
        return self.message





#test
if __name__ == '__main__':
    
    def class_extract(path):
        classes = next(os.walk( path) )
        classes=classes[1]
        return classes

    print(class_extract('data/train'))
    A=action(class_extract('data/train')) #["Stop","1","2","oui","non"])

    print(A.classes)

    A.add_action(A.classes[2])
    A.add_action(A.classes[3])
    A.add_action(A.classes[-2])
    A.add_action(A.classes[4])
    A.add_action(A.classes[-1])
    A.add_action(A.classes[0])

