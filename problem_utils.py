def opposite(action) :
    if action == 'left' :
        return 'right'
    if action == 'right' :
        return 'left'
    if action == 'up' :
        return 'down'
    if action == 'down' :
        return 'up' 

def left(action) :
    if action == 'left' :
        return 'down'
    if action == 'right' :
        return 'up'
    if action == 'up' :
        return 'left'
    if action == 'down' :
        return 'right' 

def right(action) :
    if action == 'right' :
        return 'down'
    if action == 'left' :
        return 'up'
    if action == 'up' :
        return 'right'
    if action == 'down' :
        return 'left' 

def getSuccessor(s, action) :
    
    if action == 'up' :
        return (s[0], s[1]-1)
    elif action == 'down' :
        return (s[0], s[1]+1)
    elif action == 'left' :
        return (s[0]-1, s[1])
    else :
        return (s[0]+1, s[1])
