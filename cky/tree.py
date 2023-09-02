import re
import sys

# a Tree consists of a category label 'c' and a list of child Trees 'ch'
class Tree:
    
    # obtain tree from string 
    def read(this,s):
        this.ch = []
        # a tree can be just a terminal symbol (a leaf)
        m = re.search('^ *([^ ()]+) *(.*)',s)
        if m != None:
            this.c = m.group(1)
            return m.group(2)
        # a tree can be an open paren, nonterminal symbol, subtrees, close paren
        m = re.search('^ *\( *([^ ()]*) *(.*)',s)
        if m != None:
            this.c = m.group(1)
            s = m.group(2)
            while re.search('^ *\)',s) == None:
                t = Tree()
                s = t.read(s)
                this.ch = this.ch + [t]
            return re.search('^ *\) *(.*)',s).group(1)
        return ''

    # obtain string from tree
    def __str__(this):
        if this.ch == []:
            return this.c
        s = '(' + this.c
        for t in this.ch:
            s = s + ' ' + str(t)
        return s + ')'
