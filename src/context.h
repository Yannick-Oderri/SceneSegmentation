//
// Created by ynki9 on 12/26/19.
//

#ifndef PROJECT_EDGE_CONTEXT_H
#define PROJECT_EDGE_CONTEXT_H

class AppContext;

class AppContextBuider {
public:
    AppContext* assembleAppContext();
private:
    void initializeOpenGL();
};

class AppContext {
public:
    void initialzieAppContext();
private:
    void initializeOpenGL();
    void initailzieCUDA();

};


#endif //PROJECT_EDGE_CONTEXT_H
