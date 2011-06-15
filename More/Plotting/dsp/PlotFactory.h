#ifndef __Dsp_PlotFactory_h
#define __Dsp_PlotFactory_h

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace dsp
{
    class Plot;

    class PlotFactory
    {
        public:
            PlotFactory();
            Plot* construct(string name);
            unsigned get_nplot() {return agents.size();}
            string get_name(const unsigned i);
            string get_description(const unsigned i);
            string help();

        protected:
            class Agent;
            vector<Agent*> agents;

        public:
            template<class P> class Advocate;
            void add(Agent* agent) {agents.push_back(agent);}
    };

    class PlotFactory::Agent
    {
        public:
            Agent(const char c, const string n, const string d) :
                shortcut(c), name(n), description(d) {}

            virtual ~Agent() {}
            virtual Plot* construct() = 0;
            string get_name() { return name; }
            char get_shortcut() { return shortcut; }
            string get_description() { return description; }

        protected:
            char shortcut;
            string name;
            string description;
    };

    template<class P> class PlotFactory::Advocate : public Agent
    {
        public:
            Advocate(const string _name, const string _description) :
                Agent(' ', _name, _description) {}

            Advocate(const char shortcut, const string _name,
                    const string _description) :
                Agent(shortcut, _name, _description) {}

            Plot* construct() { return new P; }
    };

}

#endif
