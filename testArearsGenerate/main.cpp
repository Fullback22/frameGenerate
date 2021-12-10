#include "arearsGenerate.h"

int main()
{
	std::vector<int> frenquncesClasses{ 1,2 };
	ArearsGenerate test{ cv::Size(512, 512) };
	test.setMainClassesParametrs();
	test.generateClasseMap();
	test.initClassesMasks();
	test.initMainImage();
}