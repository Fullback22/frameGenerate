#include "arearsGenerate.h"

int main()
{
	std::vector<int> frenquncesClasses{ 1,1 };
	ArearsGenerate test{ cv::Size(600, 400) };
	test.setMainClassesParametrs();
	test.generateClasseMap();
	test.initClassesMasks();
	test.initMainImage();
}