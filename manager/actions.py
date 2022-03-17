class action:
    def __init__(self, classes):
        self.action="No action"
        self.next_poste="No poste"
        self.next_poste_int=0
        self.classes=classes
        self.stop=0
        self.message=""

    def add_action(self, current_classe):
        last_state=self.next_poste_int
        if current_classe==self.classes[0]: #stop
            self.message="Robot stopped !"
            self.next_poste_int="STOP"
            self.stop=1

        if self.action=="Validation_step": #validation step
            if current_classe == self.classes[-1]:
                self.message=str("Let's go to the station " + self.next_poste+"!")
                self.next_poste_int=self.next_poste
                self.action="Action"
            if current_classe == self.classes[-2]:
                self.message="Action canceled !"
                self.action="Action"

        if current_classe in self.classes[1:-2]:
            self.action="Validation_step"
            self.next_poste=current_classe
            self.message=str("Do you validate station "+current_classe+ "?")

        if(last_state!=self.next_poste_int): 
            return self.message, self.next_poste_int
        return self.message, -1

# if __name__ == '__main__':