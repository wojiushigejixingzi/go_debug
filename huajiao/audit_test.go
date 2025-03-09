package huajiao

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"testing"
)

func Test_audit(t *testing.T) {
	getLog()
}

func getLog() {
	inputFile := "/Users/wwb/Desktop/audit_fix_back.txt"                     // 输入文件
	outputFile := "/Users/wwb/Desktop/audit_expansion_wall_content_logs.txt" // 输出文件

	inFile, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("无法打开文件:", err)
		return
	}
	defer inFile.Close()

	outFile, err := os.Create(outputFile)
	if err != nil {
		fmt.Println("无法创建输出文件:", err)
		return
	}
	defer outFile.Close()

	scanner := bufio.NewScanner(inFile)
	writer := bufio.NewWriter(outFile)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, `"appname":"expansion_wall_content"`) {
			writer.WriteString(line + "\n")
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("读取文件时出错:", err)
	}

	writer.Flush()
	fmt.Println("筛选完成，结果已保存至", outputFile)
}

//139 "audit_trainee" DONE
//96 "public_room_cover" DONE
//96 "public_room_avatar" DONE
//65 "personal_profile" DONE
//21 "public_room_background"。DONE
//12 "user_authentication" DONE
//11 "greet_audio" DONE
//8 "intimate_label" DONE
//6 "audio_signature" DONE
//5 "tuan_vip_name" DONE
//3 "group_management" DONE
//3 "fans_tuan" DONE
//2 "feed_crate_post"
//1 "live_user_set_introduce" DONE
//1 "expansion_wall_content" DONE

//
//once:GetHighriskChanCommand                获取感染相关信息
//once:audio_kick_stat                       语音交友提出房间主播统计
//once:auth_manage_to_dmg                    manage权限平移dmg
//once:channel_group_data                    渠道组相关数据生成命令
//once:check_server                          校验服务器是否可上线
//once:clear_live_rec_quick_task             快评任务突然暴增且为有主播评分的任务时，清理快评已经打过分的任务
//once:clear_rank_history                    清楚rank历史
//once:cmoaudit_log_stat                     通用审核日志量统计
//once:coverStat                             统计主播封面状态
//once:cover_check                           检测封面信息
//once:delete_resources                      删除资源内容
//once:dyfile_reupdate                       动态链接库切换域名ppio
//once:export_verified                       导出企业认证数据
//once:fix_live_sync                         修复后台缺失直播
//once:live_duration_stat                    审核池子每日直播时长统计
//once:menu_filter                           菜单梳理
//once:moor_web_call                         客服自动语音播报
//once:public_room_audio_cut                 截取语音房违规音频
//once:public_room_fill_apply                历史公共房补充apply表信息
//once:public_room_offline                   审核相关公共放违规下线
//once:safe_live                             安全直播统计-按照自然天纬度
//once:spam_sensitive                        反作弊敏感词库默认添加此属性个人资料(7)
//once:static_jsy_image                      统计直播鉴图图片数量
//once:sync_forbidden                        服务端封禁用户同步
//once:sync_objective_score_2_uc             同步客观分到uc属性
//once:sync_uid_ope                          同步靓号操作日志
//once:talentUserCancel                      达人认证批量取消
//once:talentUserDeal                        达人认证及转频道处理
//once:talent_data_check                     达人认证数据检查
//once:trace_stat                            后台菜单访问日志统计
//once:user_delete_test                      线上删除测试账号
//once:user_reset_nickname                   重置制定用户昵称
//once:user_score_history                    主播评级历史数据，初始化属性系统
