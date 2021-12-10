#include "arearsGenerate.h"

int main()
{
	std::vector<int> frenquncesClasses{ 1,2 };
	ArearsGenerate test{ &frenquncesClasses, cv::Size(32, 32), cv::Size(512, 512) };
	test.generateClasseMap();
	test.initClassesMasks();
	test.initMainImage();
}