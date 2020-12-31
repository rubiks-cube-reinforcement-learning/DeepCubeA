#include <vector>
#include <fstream>

/*** Abstract Environment Class ***/
class Environment {
  public:
		virtual ~Environment()=0;
    virtual Environment *getNextState(int action) const = 0;

    virtual std::vector<Environment*> getNextStates() const = 0;

    virtual std::vector<uint8_t> getState() const = 0;

    virtual bool isSolved() const = 0;

    virtual int getNumActions() const = 0;
};

/*** PuzzleN ***/
class PuzzleN: public Environment {
	private:
		static const int numActions = 4;
		uint8_t **swapZeroIdxs;

		std::vector<uint8_t> state;
		uint8_t dim;
		int numTiles;
		uint8_t zIdx;

    virtual void construct(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx);
	public:
		PuzzleN(std::vector<uint8_t> state, uint8_t dim, uint8_t zIdx);
		PuzzleN(std::vector<uint8_t> state, uint8_t dim);
		~PuzzleN();

		
    virtual PuzzleN *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

/*** LightsOut ***/
class LightsOut: public Environment {
	private:
		int **moveMat;

		std::vector<uint8_t> state;
		uint8_t dim;
		int numActions;
	public:
		LightsOut(std::vector<uint8_t> state, uint8_t dim);
		~LightsOut();

		
    virtual LightsOut *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

class Cube3: public Environment {
	private:
		static const int numActions = 12;
		static constexpr int rotateIdxs_old[12][24] = 
		{
			{2, 5, 8, 8, 7, 6, 6, 3, 0, 0, 1, 2, 38, 41, 44, 20, 23, 26, 47, 50, 53, 29, 32, 35},
			{6, 3, 0, 0, 1, 2, 2, 5, 8, 8, 7, 6, 47, 50, 53, 29, 32, 35, 38, 41, 44, 20, 23, 26},
			{11, 14, 17, 17, 16, 15, 15, 12, 9, 9, 10, 11, 45, 48, 51, 18, 21, 24, 36, 39, 42, 27, 30, 33},
			{15, 12, 9, 9, 10, 11, 11, 14, 17, 17, 16, 15, 36, 39, 42, 27, 30, 33, 45, 48, 51, 18, 21, 24},
			{20, 23, 26, 26, 25, 24, 24, 21, 18, 18, 19, 20, 45, 46, 47, 0, 1, 2, 44, 43, 42, 9, 10, 11},
			{24, 21, 18, 18, 19, 20, 20, 23, 26, 26, 25, 24, 44, 43, 42, 9, 10, 11, 45, 46, 47, 0, 1, 2},
			{29, 32, 35, 35, 34, 33, 33, 30, 27, 27, 28, 29, 38, 37, 36, 6, 7, 8, 51, 52, 53, 15, 16, 17},
			{33, 30, 27, 27, 28, 29, 29, 32, 35, 35, 34, 33, 51, 52, 53, 15, 16, 17, 38, 37, 36, 6, 7, 8},
			{38, 41, 44, 44, 43, 42, 42, 39, 36, 36, 37, 38, 18, 19, 20, 2, 5, 8, 35, 34, 33, 15, 12, 9},
			{42, 39, 36, 36, 37, 38, 38, 41, 44, 44, 43, 42, 35, 34, 33, 15, 12, 9, 18, 19, 20, 2, 5, 8},
			{47, 50, 53, 53, 52, 51, 51, 48, 45, 45, 46, 47, 29, 28, 27, 0, 3, 6, 24, 25, 26, 17, 14, 11},
			{51, 48, 45, 45, 46, 47, 47, 50, 53, 53, 52, 51, 24, 25, 26, 17, 14, 11, 29, 28, 27, 0, 3, 6}
		};

		static constexpr int rotateIdxs_new[12][24] = 
		{
			{0, 1, 2, 2, 5, 8, 8, 7, 6, 6, 3, 0, 20, 23, 26, 47, 50, 53, 29, 32, 35, 38, 41, 44},
			{0, 1, 2, 2, 5, 8, 8, 7, 6, 6, 3, 0, 20, 23, 26, 47, 50, 53, 29, 32, 35, 38, 41, 44},
			{9, 10, 11, 11, 14, 17, 17, 16, 15, 15, 12, 9, 18, 21, 24, 36, 39, 42, 27, 30, 33, 45, 48, 51},
			{9, 10, 11, 11, 14, 17, 17, 16, 15, 15, 12, 9, 18, 21, 24, 36, 39, 42, 27, 30, 33, 45, 48, 51},
			{18, 19, 20, 20, 23, 26, 26, 25, 24, 24, 21, 18, 0, 1, 2, 44, 43, 42, 9, 10, 11, 45, 46, 47},
			{18, 19, 20, 20, 23, 26, 26, 25, 24, 24, 21, 18, 0, 1, 2, 44, 43, 42, 9, 10, 11, 45, 46, 47},
			{27, 28, 29, 29, 32, 35, 35, 34, 33, 33, 30, 27, 6, 7, 8, 51, 52, 53, 15, 16, 17, 38, 37, 36},
			{27, 28, 29, 29, 32, 35, 35, 34, 33, 33, 30, 27, 6, 7, 8, 51, 52, 53, 15, 16, 17, 38, 37, 36},
			{36, 37, 38, 38, 41, 44, 44, 43, 42, 42, 39, 36, 2, 5, 8, 35, 34, 33, 15, 12, 9, 18, 19, 20},
			{36, 37, 38, 38, 41, 44, 44, 43, 42, 42, 39, 36, 2, 5, 8, 35, 34, 33, 15, 12, 9, 18, 19, 20},
			{45, 46, 47, 47, 50, 53, 53, 52, 51, 51, 48, 45, 0, 3, 6, 24, 25, 26, 17, 14, 11, 29, 28, 27},
			{45, 46, 47, 47, 50, 53, 53, 52, 51, 51, 48, 45, 0, 3, 6, 24, 25, 26, 17, 14, 11, 29, 28, 27}
		};

		std::vector<uint8_t> state;
	public:
		Cube3(std::vector<uint8_t> state);
		~Cube3();

		
    virtual Cube3 *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};

class Cube3SolvedCorners: public Environment {
	private:
		static const int numActions = 12;
		static constexpr int rotateIdxs_old[12][24] =
		{
            {33, 30, 27, 0, 3, 6, 15, 12, 9, 18, 21, 24, 38, 41, 44, 37, 43, 36, 39, 42, 42, 42, 42, 42},
            {9, 12, 15, 24, 21, 18, 27, 30, 33, 6, 3, 0, 42, 39, 36, 43, 37, 44, 41, 38, 38, 38, 38, 38},
            {47, 50, 53, 20, 23, 26, 19, 25, 18, 21, 24, 36, 39, 42, 11, 10, 9, 29, 28, 27, 27, 27, 27, 27},
            {42, 39, 36, 24, 21, 18, 25, 19, 26, 23, 20, 53, 50, 47, 27, 28, 29, 9, 10, 11, 11, 11, 11, 11},
            {45, 46, 47, 15, 12, 9, 16, 10, 17, 14, 11, 38, 37, 36, 0, 1, 2, 20, 19, 18, 18, 18, 18, 18},
            {36, 37, 38, 11, 14, 17, 10, 16, 9, 12, 15, 47, 46, 45, 20, 19, 18, 0, 1, 2, 2, 2, 2, 2},
            {34, 31, 28, 1, 4, 7, 16, 13, 10, 19, 22, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25},
            {10, 13, 16, 25, 22, 19, 28, 31, 34, 7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {46, 49, 52, 37, 40, 43, 14, 13, 12, 32, 31, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30},
            {43, 40, 37, 52, 49, 46, 30, 31, 32, 12, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14},
            {48, 49, 50, 41, 40, 39, 3, 4, 5, 23, 22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
            {39, 40, 41, 50, 49, 48, 23, 22, 21, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}
		};

		static constexpr int rotateIdxs_new[12][24] =
		{
            {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 37, 38, 39, 41, 42, 43, 44, 44, 44, 44, 44},
            {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 37, 38, 39, 41, 42, 43, 44, 44, 44, 44, 44},
            {9, 10, 11, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 36, 39, 42, 47, 50, 53, 53, 53, 53, 53},
            {9, 10, 11, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 36, 39, 42, 47, 50, 53, 53, 53, 53, 53},
            {0, 1, 2, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 36, 37, 38, 45, 46, 47, 47, 47, 47, 47},
            {0, 1, 2, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 36, 37, 38, 45, 46, 47, 47, 47, 47, 47},
            {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34},
            {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34},
            {12, 13, 14, 30, 31, 32, 37, 40, 43, 46, 49, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52},
            {12, 13, 14, 30, 31, 32, 37, 40, 43, 46, 49, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52},
            {3, 4, 5, 21, 22, 23, 39, 40, 41, 48, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
            {3, 4, 5, 21, 22, 23, 39, 40, 41, 48, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50}
		};

		std::vector<uint8_t> state;
	public:
		Cube3SolvedCorners(std::vector<uint8_t> state);
		~Cube3SolvedCorners();


    virtual Cube3SolvedCorners *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};


class Cube4: public Environment {
	private:
		static const int numActions = 24;
		static const std::vector<std::vector<int> > rotateIdxs_old;
		static const std::vector<std::vector<int> > rotateIdxs_new;

		std::vector<uint8_t> state;
	public:
		Cube4(std::vector<uint8_t> state);
		~Cube4();
		
    virtual Cube4 *getNextState(int action) const;

    virtual std::vector<Environment*> getNextStates() const;

    virtual std::vector<uint8_t> getState() const;

    virtual bool isSolved() const;

    virtual int getNumActions() const;
};


