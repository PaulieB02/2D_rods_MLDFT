// Periodic square-lattice grand-canonical hard rods, no external field.
// Each rod has activity exp(mu); mu is beta*mu and is common to both species.
// Build: g++ -O3 -march=native -std=c++17 -Wall -Wextra rods_gcmc.cpp -o rods_gcmc
// The default exact conditional line updates follow Kundu et al., PRE 87,
// 032103 (2013), Sec. II. No high-density flip moves are needed here.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

struct RNG {
    uint64_t s[4];
    static uint64_t rot(uint64_t x, int k) { return (x << k) | (x >> (64-k)); }
    explicit RNG(uint64_t seed) {
        for (auto &v:s) {
            uint64_t z=(seed+=0x9e3779b97f4a7c15ULL);
            z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
            z=(z^(z>>27))*0x94d049bb133111ebULL;
            v=z^(z>>31);
        }
    }
    uint64_t next() { // xoshiro256++
        uint64_t result=rot(s[0]+s[3],23)+s[0], t=s[1]<<17;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3];
        s[2]^=t; s[3]=rot(s[3],45); return result;
    }
    double uniform() { return (next()>>11)*0x1.0p-53; }
    uint64_t integer(uint64_t bound) {
        // Unbiased multiply-and-reject bounded integer (no modulo bias).
        uint64_t x=next(); __uint128_t m=(__uint128_t)x*bound;
        uint64_t low=(uint64_t)m;
        if (low<bound) {
            uint64_t threshold=-bound%bound;
            while (low<threshold) { x=next(); m=(__uint128_t)x*bound; low=(uint64_t)m; }
        }
        return (uint64_t)(m>>64);
    }
};

struct Moments {
    uint64_t n=0, empty=0;
    long double eta=0, eta2=0, abs_s=0, s2=0, s4=0, s=0;
    void add(int n0, int n1, int k, int volume) {
        double e=double(k)*(n0+n1)/volume;
        double q=n0+n1 ? double(n0-n1)/(n0+n1) : 0.0;
        ++n; empty+=(n0+n1==0); eta+=e; eta2+=e*e;
        abs_s+=std::abs(q); s2+=q*q; s4+=q*q*q*q; s+=q;
    }
};

class Rods {
public:
    const int L, k, V;
    const double mu, activity;
    RNG rng;
    std::vector<uint8_t> occ; // 0 empty, 1 vertical, 2 horizontal
    std::array<std::vector<int>,2> rods; // anchor sites, swap-and-pop deletion
    std::vector<int> cells, line_sites;
    std::vector<double> p_rod;
    double p_ring_empty;

    Rods(int length, int rod_length, double chemical_potential, uint64_t seed)
        :L(length),k(rod_length),V(L*L),mu(chemical_potential),activity(std::exp(mu)),
         rng(seed),occ(V,0),cells(2*V*k),line_sites(4*V),p_rod(L+1,0) {
        if (L<=k || k<1) throw std::runtime_error("require 1 <= k < L");
        for (int d=0;d<2;++d) {
            rods[d].reserve(V/k);
            for (int a=0;a<V;++a) for (int j=0;j<k;++j) {
                int y=a/L,x=a%L;
                cells[(d*V+a)*k+j]=d ? y*L+(x+j)%L : ((y+j)%L)*L+x;
            }
            for (int line=0;line<L;++line) for (int j=0;j<2*L;++j)
                line_sites[(d*L+line)*2*L+j]=d ? line*L+j%L : (j%L)*L+line;
        }
        // log Z_open(n) = log[Z_open(n-1) + z Z_open(n-k)].
        std::vector<double> logz(L+1,0);
        for (int n=k;n<=L;++n) {
            double a=logz[n-1], b=mu+logz[n-k], h=std::max(a,b);
            logz[n]=h+std::log1p(std::exp(-std::abs(a-b)));
            p_rod[n]=std::exp(b-logz[n]);
        }
        // On an obstacle-free ring, site 0 is either empty or covered by
        // one of k possible rod offsets: Z_ring = Z_open(L-1)+k*z*Z_open(L-k).
        double delta=std::log(double(k))+mu+logz[L-k]-logz[L-1];
        p_ring_empty=1/(1+std::exp(delta));
    }
    int n(int d) const { return int(rods[d].size()); }
    bool fits(int d,int anchor) const {
        const int *c=&cells[(d*V+anchor)*k];
        for (int j=0;j<k;++j) if (occ[c[j]]) return false;
        return true;
    }
    void put(int d,int anchor) {
        const int *c=&cells[(d*V+anchor)*k];
        for (int j=0;j<k;++j) occ[c[j]]=d+1;
        rods[d].push_back(anchor);
    }
    void erase(int d,int index) {
        const int *c=&cells[(d*V+rods[d][index])*k];
        for (int j=0;j<k;++j) occ[c[j]]=0;
        rods[d][index]=rods[d].back(); rods[d].pop_back();
    }
    void initial(const std::string &kind) {
        if (kind=="empty") return;
        if (kind!="vertical" && kind!="horizontal") throw std::runtime_error("invalid initial state");
        int d=kind=="horizontal";
        for (int line=0;line<L;++line) for (int pos=0;pos+k<=L;pos+=k)
            put(d,line_sites[(d*L+line)*2*L+pos]);
    }
    void sample_open(int d,const int *sites,int pos,int length) {
        while (length>=k) {
            if (rng.uniform()<p_rod[length]) { put(d,sites[pos]); pos+=k; length-=k; }
            else { ++pos; --length; }
            if (pos>=L) pos-=L;
        }
    }
    void heatbath_species(int d) {
        while (!rods[d].empty()) erase(d,n(d)-1);
        uint8_t obstacle=2-d;
        for (int line=0;line<L;++line) {
            const int *sites=&line_sites[(d*L+line)*2*L];
            int blocked=0;
            while (blocked<L && occ[sites[blocked]]!=obstacle) ++blocked;
            if (blocked==L) {
                if (rng.uniform()<p_ring_empty) sample_open(d,sites,1,L-1);
                else {
                    int offset=int(rng.integer(k));
                    int head=offset ? L-offset : 0;
                    put(d,sites[head]);
                    int start=head+k; if (start>=L) start-=L;
                    sample_open(d,sites,start,L-k);
                }
            } else {
                int length=0,start=0;
                for (int j=1;j<=L;++j) {
                    int pos=blocked+j; if (pos>=L) pos-=L;
                    if (occ[sites[pos]]==obstacle) {
                        if (length>=k) sample_open(d,sites,start,length);
                        length=0;
                    } else {
                        if (!length) start=pos;
                        ++length;
                    }
                }
            }
        }
    }
    void sweep(bool heatbath) {
        if (heatbath) {
            // Random phase order also makes the full-sweep mixture reversible.
            int first=int(rng.integer(2)); heatbath_species(first); heatbath_species(1-first);
        } else {
            const double proposal_weight=2.0*V*activity;
            for (int step=0;step<V;++step) {
                int total=n(0)+n(1);
                if (rng.integer(2)) {
                    int d=int(rng.integer(2)),a=int(rng.integer(V));
                    if (fits(d,a) && (proposal_weight>=total+1 || rng.uniform()*(total+1)<proposal_weight)) put(d,a);
                } else if (total) {
                    int i=int(rng.integer(total)),d=i>=n(0);
                    if (d) i-=n(0);
                    if (total>=proposal_weight || rng.uniform()*proposal_weight<total) erase(d,i);
                }
            }
        }
    }
    void verify() const {
        std::vector<uint8_t> expected(V,0);
        for (int d=0;d<2;++d) for (int a:rods[d]) for (int j=0;j<k;++j) {
            int site=cells[(d*V+a)*k+j];
            if (expected[site]) throw std::runtime_error("overlapping rods");
            expected[site]=d+1;
        }
        if (expected!=occ) throw std::runtime_error("rod list / occupancy mismatch");
    }
};

std::array<long double,4> exact_small(int L,int k,double mu) {
    Rods model(L,k,mu,1);
    std::vector<std::vector<std::pair<uint64_t,int>>> covering(L*L);
    for (int d=0;d<2;++d) for (int a=0;a<L*L;++a) {
        uint64_t mask=0;
        for (int j=0;j<k;++j) mask|=1ULL<<model.cells[(d*L*L+a)*k+j];
        for (int j=0;j<L*L;++j) if (mask&(1ULL<<j)) covering[j].push_back({mask,d});
    }
    long double z=0,eta=0,abs_s=0,s2=0;
    std::function<void(uint64_t,int,int,long double)> visit;
    visit=[&](uint64_t available,int n0,int n1,long double weight) {
        if (!available) {
            int N=n0+n1; long double q=N ? (long double)(n0-n1)/N : 0;
            z+=weight; eta+=weight*k*N/(L*L); abs_s+=weight*std::abs(q); s2+=weight*q*q; return;
        }
        int first=__builtin_ctzll(available);
        visit(available&~(1ULL<<first),n0,n1,weight);
        for (auto [mask,d]:covering[first]) if ((available&mask)==mask)
            visit(available^mask,n0+(d==0),n1+(d==1),weight*std::exp(mu));
    };
    visit((1ULL<<(L*L))-1,0,0,1);
    return {eta/z,abs_s/z,s2/z-(abs_s/z)*(abs_s/z),z};
}

void self_test() {
    for (double mu:{-1.0,0.0,0.8}) {
        auto exact=exact_small(4,2,mu);
        for (bool heatbath:{true,false}) {
            Rods model(4,2,mu,19281+uint64_t((mu+2)*100)); Moments m;
            for (int t=0;t<220000;++t) {
                model.sweep(heatbath);
                if (t%1000==0) model.verify();
                if (t>=20000) m.add(model.n(0),model.n(1),2,16);
            }
            long double e=m.eta/m.n,q=m.abs_s/m.n,v=m.s2/m.n-q*q;
            std::cerr<<(heatbath?"heatbath":"local")<<" mu="<<mu
                     <<" eta="<<double(e)<<" exact="<<double(exact[0])
                     <<" absS="<<double(q)<<" exact="<<double(exact[1])
                     <<" variance="<<double(v)<<" exact="<<double(exact[2])<<"\n";
            if (std::abs(e-exact[0])>0.006 || std::abs(q-exact[1])>0.006 || std::abs(v-exact[2])>0.006)
                throw std::runtime_error("small-lattice exact-enumeration test failed");
        }
    }
    Rods monomers(8,1,-0.4,918231); Moments m;
    for (int t=0;t<60000;++t) { monomers.sweep(true); if (t>=10000) m.add(monomers.n(0),monomers.n(1),1,64); }
    double expected=2*std::exp(-0.4)/(1+2*std::exp(-0.4));
    if (std::abs(double(m.eta/m.n)-expected)>0.003) throw std::runtime_error("ideal monomer test failed");
    std::cerr<<"All exact-enumeration, ideal-monomer and occupancy tests passed.\n";
}

int main(int argc,char **argv) try {
    int L=100,k=8; uint64_t burn=5000,sweeps=20000,block=500,stride=1,seed=1,verify_every=0;
    double mu=0; std::string output="blocks.csv",initial="empty",sampler="heatbath";
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        if (key=="--self-test") { self_test(); return 0; }
        if (key=="--help") {
            std::cout<<"--L 100 --k 8 --mu 0 --burn 5000 --sweeps 20000 --block 500 --stride 1\n"
                     <<"--seed 1 --initial empty|vertical|horizontal --sampler heatbath|local\n"
                     <<"--output blocks.csv --verify-every 0 --self-test\n"; return 0;
        }
        if (i+1>=argc) throw std::runtime_error("missing value for "+key);
        std::string v=argv[++i];
        if (key=="--L") L=std::stoi(v); else if (key=="--k") k=std::stoi(v);
        else if (key=="--mu") mu=std::stod(v); else if (key=="--burn") burn=std::stoull(v);
        else if (key=="--sweeps") sweeps=std::stoull(v); else if (key=="--block") block=std::stoull(v);
        else if (key=="--stride") stride=std::stoull(v); else if (key=="--seed") seed=std::stoull(v);
        else if (key=="--initial") initial=v; else if (key=="--sampler") sampler=v;
        else if (key=="--output") output=v; else if (key=="--verify-every") verify_every=std::stoull(v);
        else throw std::runtime_error("unknown option "+key);
    }
    if (L<2 || L>2000 || k<1 || k>=L || !std::isfinite(mu) || std::abs(mu)>100 || !block || !stride || !sweeps || block%stride || sweeps%block)
        throw std::runtime_error("invalid parameters (require stride divides block divides sweeps)");
    if (sampler!="heatbath" && sampler!="local") throw std::runtime_error("invalid sampler");
    auto t0=std::chrono::steady_clock::now();
    Rods model(L,k,mu,seed); model.initial(initial); model.verify();
    for (uint64_t t=0;t<burn;++t) { model.sweep(sampler=="heatbath"); if (verify_every && t%verify_every==0) model.verify(); }
    std::ofstream out(output);
    if (!out) throw std::runtime_error("cannot open "+output);
    out<<"L,k,mu,seed,initial,sampler,burn,block_sweeps,stride,block,samples,empty,sum_eta,sum_eta2,sum_abs_s,sum_s2,sum_s4,sum_s,sign_changes\n"<<std::setprecision(17);
    Moments total; int last_sign=0; uint64_t changes=0;
    for (uint64_t b=0;b<sweeps/block;++b) {
        Moments m; uint64_t bchanges=0;
        for (uint64_t t=0;t<block;++t) {
            model.sweep(sampler=="heatbath");
            if (verify_every && (b*block+t)%verify_every==0) model.verify();
            if ((t+1)%stride==0) {
                m.add(model.n(0),model.n(1),k,L*L); total.add(model.n(0),model.n(1),k,L*L);
                int sign=(model.n(0)>model.n(1))-(model.n(0)<model.n(1));
                if (sign && last_sign && sign!=last_sign) { ++bchanges; ++changes; }
                if (sign) last_sign=sign;
            }
        }
        out<<L<<','<<k<<','<<mu<<','<<seed<<','<<initial<<','<<sampler<<','<<burn<<','<<block<<','<<stride<<','<<b<<','<<m.n<<','<<m.empty
           <<','<<double(m.eta)<<','<<double(m.eta2)<<','<<double(m.abs_s)<<','<<double(m.s2)<<','<<double(m.s4)<<','<<double(m.s)<<','<<bchanges<<'\n';
        out.flush();
    }
    model.verify();
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    long double q=total.abs_s/total.n;
    std::cerr<<std::setprecision(8)<<"mu="<<mu<<" eta="<<double(total.eta/total.n)<<" absS="<<double(q)
             <<" var="<<double(total.s2/total.n-q*q)<<" sign_changes="<<changes<<" seconds="<<elapsed<<'\n';
    return 0;
} catch (const std::exception &e) { std::cerr<<"error: "<<e.what()<<'\n'; return 1; }
